# Third-party notices

## Our code: licence pending owner decision

No licence is asserted for the publication authors' own code in this folder
(runtime frontend, scripts, docs, configs). Do not assume an open-source
licence. (Returned as a blocker in the delivery note.)

## DeepFilterNet (upstream)

- The runtime vendors an isolated fork of `libDF` (`runtime/libDF/`) and the
  training scripts build on the upstream DeepFilterNet v0.5.6 Python source.
- Upstream repository: https://github.com/Rikorose/DeepFilterNet
- Licence per upstream `runtime/libDF/Cargo.toml`: `MIT/Apache-2.0` (dual-licensed).
- No `LICENSE-MIT` / `LICENSE-APACHE` texts were present in the isolated
  workspace copy; obtain the licence texts from the upstream repository before
  redistributing binaries, and retain this attribution.
- The upstream **pretrained DeepFilterNet3 weights** are a separate artifact
  from our fine-tune: they initialise training only and are not published here.
  Our **fine-tuned ONNX bundle** is likewise not published (no affirmative
  publication right found; redistribution clearance outstanding).

## Datasets (not published; needed to reproduce training)

- **VoiceBank** clean speech (`clean_trainset_28spk` family, 28 training speakers
  used): proprietary research corpus by Veaux et al. Obtain from the original
  distributor and follow its terms; no licence text is bundled here.
- **SESA** noise (Sound Events for Separation and Analysis; SHA-256-pinned zip
  in `training/config.env`): follow the SESA release terms; 340 train clips used.
- **ESC-50** environmental sounds (folds 4-5 excluded from our splits by design):
  follow the ESC-50 release terms.
- Archive hashes, source lists, and split provenance live in the private training
  snapshot (`docs/evidence/training_snapshot_verification.json` records its
  integrity); raw audio is not redistributed here.

## Rust crates

Pinned in `runtime/Cargo.lock`. Notable direct dependencies of `runtime/`:
`anyhow`, `clap`, `ndarray`, `serde_json`, `sha2`, `flate2`, `tar`, `rust-ini`,
plus the transitive dependency tree of the vendored `libDF` (`tract-*`,
`rustfft`, `hound`, and others) — each under its own licence as recorded in
its upstream repository and `Cargo.lock` metadata. Review crate licences before
redistributing binaries.

## Python packages

`numpy`, `scipy`, `sounddevice` (+ PortAudio system library), `soundfile`
(+ libsndfile), `psutil`, and (for training/evaluation only) `torch`, `onnx`,
`onnxruntime`, `pystoi`, `pesq`, `soundfile`, `h5py` — each under its own
licence; install from PyPI and follow their terms.