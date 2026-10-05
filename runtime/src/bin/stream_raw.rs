//! Stateful DeepFilterNet v0.5.6 processor over a raw PCM pipe.
//! stdin: mono f32le at 48 kHz, exactly 480 samples per frame.
//! stdout: same format, one output frame for each complete input frame.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::collections::BTreeSet;

use anyhow::{Context, Result};
use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::Array2;
use sha2::{Digest, Sha256};

fn check_bundle(bytes: &[u8]) -> Result<()> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(bytes));
    let expected: BTreeSet<String> = ["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"]
        .iter().map(|s| s.to_string()).collect();
    let mut found = BTreeSet::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let name = entry.path()?.to_string_lossy().to_string();
        if !entry.header().entry_type().is_file() || !expected.contains(&name) || !found.insert(name.clone()) {
            anyhow::bail!("invalid or duplicate model bundle member: {name}");
        }
        if entry.size() == 0 || entry.size() > 64 * 1024 * 1024 {
            anyhow::bail!("invalid model member size: {name}");
        }
        if name == "config.ini" {
            let config = ini::Ini::read_from(&mut entry)?;
            let df = config.section(Some("df")).context("missing df config")?;
            let train = config.section(Some("train")).context("missing train config")?;
            if df.get("sr") != Some("48000") || df.get("hop_size") != Some("480")
                || train.get("model") != Some("deepfilternet3") {
                anyhow::bail!("bundle must be DeepFilterNet3, 48 kHz, 480-sample hop");
            }
        }
    }
    if found != expected { anyhow::bail!("incomplete model bundle"); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle(names: &[&str], rate: u32) -> Vec<u8> {
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for name in names {
            let data = if *name == "config.ini" {
                format!("[train]\nmodel=deepfilternet3\n[df]\nsr={rate}\nhop_size=480\n").into_bytes()
            } else {
                vec![1, 2, 3]
            };
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, &data[..]).unwrap();
        }
        archive.into_inner().unwrap().finish().unwrap()
    }

    #[test]
    fn validates_complete_bundle_and_rejects_bad_identity() {
        let members = ["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"];
        assert!(check_bundle(&bundle(&members, 48_000)).is_ok());
        assert!(check_bundle(&bundle(&members, 16_000)).is_err());
        assert!(check_bundle(&bundle(&members[..3], 48_000)).is_err());
        assert!(check_bundle(&bundle(&["enc.onnx", "enc.onnx", "df_dec.onnx", "config.ini"], 48_000)).is_err());
        assert!(check_bundle(&bundle(&["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini", "extra.txt"], 48_000)).is_err());
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut pf_beta = None;
    let mut bundle: Option<PathBuf> = None;
    let mut parity_mode = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pf-beta" => {
                let value: f32 = args.next().context("--pf-beta needs a value")?.parse()?;
                if !value.is_finite() || !(0.0..=0.2).contains(&value) {
                    anyhow::bail!("--pf-beta must be finite and between 0 and 0.2");
                }
                pf_beta = Some(value);
            }
            "--model-bundle" => {
                bundle = Some(PathBuf::from(args.next().context("--model-bundle needs a path")?));
            }
            "--parity-mode" => parity_mode = true,
            _ => anyhow::bail!("unknown argument: {arg}"),
        }
    }
    let mut runtime = RuntimeParams::default()
        .with_atten_lim(100.0)
        .with_thresholds(-15.0, 35.0, 35.0)
        .with_mask_reduce(ReduceMask::MAX);
    if let Some(value) = pf_beta {
        runtime = runtime.with_post_filter(value);
    }
    if parity_mode {
        if pf_beta.is_some() { anyhow::bail!("parity mode requires no postfilter"); }
        runtime = runtime.with_thresholds(-1.0e9, 1.0e9, 1.0e9);
    }
    let path = match bundle {
        Some(path) => path,
        None => anyhow::bail!("--model-bundle <bundle.tar.gz> is required (no model is embedded in this publication build)"),
    };
    let bytes = std::fs::read(&path).context("cannot read model bundle")?;
    check_bundle(&bytes)?;
    let model_hash = format!("{:x}", Sha256::digest(&bytes));
    let (params, model_mode) = (DfParams::new(path)?, "custom");
    let mut model = DfTract::new(params, &runtime)?;
    let hop = model.hop_size;
    if model.sr != 48_000 || hop != 480 {
        anyhow::bail!("model requires {} Hz/{hop} hop, expected 48000/480", model.sr);
    }
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let mut input_bytes = vec![0_u8; hop * 4];
    let mut output_bytes = vec![0_u8; hop * 4];
    let mut input = Array2::<f32>::zeros((1, hop));
    let mut output = Array2::<f32>::zeros((1, hop));
    let mut frames = 0_u64;
    eprintln!(
        "ready sample_rate=48000 hop_samples={hop} model_delay_samples={} model_mode={model_mode} model_sha256={model_hash} parity_mode={parity_mode}",
        model.fft_size - hop + model.lookahead * hop
    );
    loop {
        let mut received = 0;
        while received < input_bytes.len() {
            let n = stdin.read(&mut input_bytes[received..])?;
            if n == 0 {
                if received != 0 {
                    anyhow::bail!("partial input frame at EOF: {received} bytes");
                }
                eprintln!("processed_frames={frames}");
                return Ok(());
            }
            received += n;
        }
        for (sample, bytes) in input.iter_mut().zip(input_bytes.chunks_exact(4)) {
            *sample = f32::from_le_bytes(bytes.try_into().context("invalid f32 sample")?);
            if !sample.is_finite() { anyhow::bail!("non-finite input sample"); }
        }
        if parity_mode {
            model.process_continuous(input.view(), output.view_mut())?;
        } else {
            model.process(input.view(), output.view_mut())?;
        }
        for (bytes, &sample) in output_bytes.chunks_exact_mut(4).zip(output.iter()) {
            bytes.copy_from_slice(&sample.to_le_bytes());
        }
        stdout.write_all(&output_bytes)?;
        stdout.flush()?;
        frames += 1;
    }
}
