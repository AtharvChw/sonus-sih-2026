#!/usr/bin/env python3
"""06b_smoke.py — REAL CUDA smoke test for one run dir (Kaggle GPU only).

Uses ONLY pinned v0.5.6 upstream APIs (same call shapes as df/train.py and
df/enhance.py). Proves, against real HDF5 data and real checkpoints:
  S1  imports + CUDA device + GPU name
  S2  HDF5 databases open, sr attr == 48000, one entry readable
  S3  run config loads; model builds; checkpoint loads (reports epoch); params > 0
  S4  one REAL batch: forward -> finite loss -> backward -> finite nonzero grads ->
      optimizer step changes params -> write_cp -> fresh model + read_cp reload ->
      params identical
  S5  waveform inference on the fixed probe via init_df + explicit read_cp -> finite output WAV
      of matching length, saved + hashed
Any failure prints FAIL + reason and exits 1. Writes smoke_report_<tag>.json next
to the run dir. This is NOT training and NEVER touches the frozen 280-pair test.

Usage:
  python3 06b_smoke.py --run-dir runs/impulse_weighted --data-dir data \\
      --dataset-cfg data/dataset.cfg --probe <fixed 48k mono wav> --epoch 0 \\
      --tag base --batch-size 2
  (epoch 0 = starting weights; epoch best = fine-tuned best after training)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import torch


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(detail)) if detail else ""))
    if not cond:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-cfg", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--epoch", required=True, help="'0' for starting weights, 'best' after training")
    ap.add_argument("--tag", required=True, help="e.g. base | best")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run_dir)
    report = {"run": run.name, "epoch": args.epoch, "tag": args.tag}

    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    # input fingerprints: C7 re-checks these and blocks training on drift.
    # NOTE: HDF5 bytes are deliberately NOT fingerprinted (04 rebuilds them per session;
    # container metadata may differ). lists/split_manifest.csv IS fingerprinted - identical
    # manifest + identical source archives (02 hash gate) means identical HDF5 content.
    dcfg = Path(args.dataset_cfg)
    report["inputs"] = {
        "dataset_cfg": str(dcfg),
        "dataset_cfg_sha256": _sha(dcfg),
        "split_manifest_sha256": _sha(Path(args.data_dir).parent / "lists" / "split_manifest.csv"),
        "run_config_sha256": _sha(run / "config.ini"),
        "model_0_sha256": _sha(run / "checkpoints" / "model_0.ckpt"),
        "hdf5_sha256": {p.name: _sha(p) for p in sorted(Path(args.data_dir).glob("*.hdf5"))},
    }

    # ---- S1: imports + CUDA ----
    check("S1 torch CUDA available", torch.cuda.is_available())
    from df.config import config
    config.load(str(run / "config.ini"))
    from df.modules import get_device  # noqa: E402
    dev = get_device()
    check("S1 device is cuda", dev.type == "cuda", str(dev))
    gname = torch.cuda.get_device_name(0)
    report["gpu"] = gname
    print("GPU:", gname)
    import df  # noqa: E402,F401
    import libdf  # noqa: E402,F401
    import libdfdata  # noqa: E402,F401
    print("df/libdf/libdfdata imports OK")

    # ---- S2: HDF5 databases ----
    from libdfdata import PytorchDataLoader as DataLoader  # noqa: E402
    for f in ("train_speech.hdf5", "train_sesa_noise.hdf5", "train_esc_noise.hdf5"):
        p = Path(args.data_dir) / f
        check("S2 present " + f, p.is_file())
        with h5py.File(p) as h:
            kind = "speech" if "speech" in f else "noise"
            check("S2 48kHz nonempty " + f,
                  h.attrs["sr"] == 48000 and len(h[kind]) > 0,
                  "sr=%s items=%d" % (h.attrs["sr"], len(h[kind])))

    # ---- S3: config + model + checkpoint load ----
    from df.config import config  # noqa: E402
    from df.model import ModelParams  # noqa: E402
    from df.checkpoint import load_model, write_cp, read_cp  # noqa: E402
    from df.utils import get_norm_alpha  # noqa: E402
    from libdf import DF  # noqa: E402
    p = ModelParams()
    state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size,
               nb_bands=p.nb_erb, min_nb_erb_freqs=p.min_nb_freqs)
    ckpt_dir = run / "checkpoints"
    # NOTE: load_model defaults to epoch "latest". Pre-training that IS model_0;
    # post-training it is the last epoch (not necessarily "best"). S3/S4 prove the
    # load+step machinery on real weights either way; S5 selects the epoch explicitly.
    model, epoch = load_model(str(ckpt_dir), state, jit=False,
                              mask_only=False, train_df_only=False)
    n_params = sum(v.numel() for v in model.parameters())
    check("S3 checkpoint loads, params>0", n_params > 0,
          "loaded_epoch=%s params=%d" % (epoch, n_params))
    report["loaded_epoch"] = epoch
    report["params"] = n_params
    model.to(dev)

    # ---- S4: one real batch, full step, save+reload ----
    loader = DataLoader(
        ds_dir=args.data_dir, ds_config=args.dataset_cfg, sr=p.sr,
        batch_size=args.batch_size, batch_size_eval=args.batch_size,
        num_workers=2, pin_memory=True,
        max_len_s=5.0, fft_size=p.fft_size, hop_size=p.hop_size,
        nb_erb=p.nb_erb, nb_spec=p.nb_df, norm_alpha=get_norm_alpha(log=False),
        p_reverb=0.0, p_bw_ext=0.0, p_clipping=0.0, p_zeroing=0.0,
        p_air_absorption=0.0, p_interfer_sp=0.0, prefetch=4,
        overfit=False, seed=26052, min_nb_erb_freqs=p.min_nb_freqs,
        log_timings=False,
        global_sampling_factor=1.0,
    )
    batch = next(iter(loader.iter_epoch("train", 26052)))
    check("S4 batch fields",
          batch.feat_spec is not None and batch.feat_erb is not None,
          "keys=feat_erb/feat_spec/noisy/speech/snr")
    from df.utils import as_real  # noqa: E402
    from df.loss import Istft, Loss  # noqa: E402
    feat_erb = batch.feat_erb.to(dev)
    feat_spec = as_real(batch.feat_spec.to(dev))
    noisy = batch.noisy.to(dev)
    clean = batch.speech.to(dev)
    snrs = batch.snr.to(dev)
    model.train()
    enh, m, lsnr, _ = model.forward(spec=as_real(noisy), feat_erb=feat_erb, feat_spec=feat_spec)
    istft = Istft(p.fft_size, p.hop_size,
                  torch.as_tensor(state.fft_window().copy())).to(dev)
    loss_fn = Loss(state, istft).to(dev)
    err = loss_fn.forward(clean, noisy, enh, m, lsnr, snrs=snrs)
    loss_val = float(err.item())
    check("S4 finite loss", np.isfinite(loss_val), loss_val)
    report["loss"] = loss_val
    lr = 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()
    err.backward()
    gnorms = [float(v.grad.norm()) for v in model.parameters() if v.grad is not None]
    check("S4 grads finite+nonzero",
          gnorms and all(np.isfinite(g) for g in gnorms) and max(gnorms) > 0,
          "max_grad=%.3e" % max(gnorms, default=0.0))
    report["max_grad"] = max(gnorms)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    before = [v.detach().clone() for v in model.parameters()]
    opt.step()
    delta = max(float((a - b).abs().max()) for a, b in
                zip(model.parameters(), before))
    check("S4 optimizer step changes params", delta > 0, "max_delta=%.3e" % delta)
    report["param_delta"] = delta
    smoke_ckpt = ckpt_dir / "smoke_test_0.ckpt"
    write_cp(model, "smoke_test", str(ckpt_dir), 0)
    check("S4 checkpoint written", smoke_ckpt.is_file())
    model2, _ = load_model(str(ckpt_dir), state, jit=False,
                           mask_only=False, train_df_only=False)
    # load_model loads latest/best, not our smoke file: load it explicitly
    from df.checkpoint import read_cp as _rc  # noqa: E402
    _rc(model2, "smoke_test", str(ckpt_dir), log=False)
    saved = model.state_dict()
    reloaded = model2.state_dict()
    same = saved.keys() == reloaded.keys() and all(
        torch.equal(saved[k].cpu(), reloaded[k].cpu()) for k in saved)
    check("S4 reload identical", same)
    smoke_ckpt.unlink(missing_ok=True)

    # ---- S5: waveform inference on fixed probe ----
    from df.enhance import init_df, enhance  # noqa: E402
    from torch.nn.functional import interpolate  # noqa: E402
    ep = 0 if str(args.epoch) == "0" else args.epoch
    # Upstream init_df falsely treats epoch 0 as a missing checkpoint. Initialize
    # first, then load the selected checkpoint via the verified public reader.
    emodel, df_state, _suffix = init_df(str(run), epoch=None, config_allow_defaults=True)
    inference_epoch = read_cp(emodel, "model", str(ckpt_dir), epoch=ep)
    check("S5 selected checkpoint loaded", inference_epoch is not None
          and (not isinstance(ep, int) or inference_epoch == ep), inference_epoch)
    report["inference_epoch"] = inference_epoch
    wav, sr = sf.read(args.probe, dtype="float32", always_2d=False)
    check("S5 probe readable", wav.ndim == 1 and len(wav) > sr,
          "samples=%d sr=%d" % (len(wav), sr))
    t = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)  # enhance wants [C, T]
    if sr != df_state.sr():
        t = interpolate(t.unsqueeze(0), scale_factor=df_state.sr() / sr,
                        mode="linear").squeeze(0)
        print("S5 resampled probe %d -> %d Hz (pure torch)" % (sr, df_state.sr()))
    out = enhance(emodel, df_state, t)
    out = np.asarray(out.detach().cpu()).reshape(-1)
    check("S5 output finite+nontrivial, length sane",
          np.all(np.isfinite(out)) and len(out) > 0 and np.max(np.abs(out)) > 1e-6
          and abs(len(out) - t.shape[-1]) <= 4096,
          "out_len=%d in_len=%d peak=%.4f"
          % (len(out), t.shape[-1], float(np.max(np.abs(out)))))
    out_path = run / ("smoke_out_%s.wav" % args.tag)
    sf.write(str(out_path), out.astype(np.float32), df_state.sr())
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    report["smoke_wav"] = {"path": str(out_path), "sha256": h, "sr": df_state.sr()}
    print("smoke wav:", out_path, h[:16])

    rep_path = run / ("smoke_report_%s.json" % args.tag)
    report["ok"] = True
    report["schema_version"] = 1
    rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SMOKE ALL PASS ::", rep_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
