# Tests

The only test evidence this publication asserts is the existing Rust
bundle-validation unit test inside `runtime/src/bin/stream_raw.rs`
(accepts a complete valid bundle; rejects wrong sample rate, missing members,
duplicates, and extras).

Run it from `runtime/`:

```bash
cargo test --release
```

Full proof is preserved in `docs/evidence/rust_linux_tests.log`.
Waveform parity is a gate, not a unit test — run it with
`evaluation/verify_custom_parity.py` (see `docs/evaluation.md`).
No tests were invented to reach any count.