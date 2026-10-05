# Decisions (short ablation)

## Selected: `no_oversampling` (six-epoch DeepFilterNet3 fine-tune, epoch-6 best)

480-mixture development validation, raw outputs, identical mixtures per system (rounded for readability; same convention as the README — evidence JSONs keep full precision):

| System | Output SNR (dB) | SI-SNR (dB) | STOI | PESQ-WB |
|---|---|---:|---:|---:|
| Bypass (no enhancement) | 5.00 | 5.05 | 0.838 | 1.50 |
| Upstream pretrained DFN3 | 16.37 | 16.50 | 0.917 | 3.00 |
| `impulse_weighted` (SESA train x2.0) | 17.01 | 17.15 | 0.917 | 3.01 |
| `no_oversampling` (SESA train x1.0, selected) | 16.99 | 17.11 | 0.918 | 3.03 |

## Why `no_oversampling` over `impulse_weighted`

- Highest STOI (0.918) and PESQ-WB (3.03) of the comparison.
- Fewest material speech regressions vs pretrained: STOI drop > 0.02 in 12/480
  (vs 20/480 weighted) and PESQ drop > 0.1 in 92/480 (vs 95/480 weighted);
  SNR drop > 1 dB in 3/480 (vs 2/480 weighted).
- SNR essentially tied (+0.62 dB vs +0.64 dB mean delta over pretrained,
  overlapping exploratory CIs).

The choice is a speech-preservation tradeoff for runtime testing, not a claim of
universal superiority: STOI/PESQ gain CIs cross zero, and 92/480 PESQ regressions
remain under review. Higher training loss (0.6613889 vs 0.6610152) did not predict
the perceptual tradeoff; selection used validation metrics, never the sealed test.

## Why not the alternatives

- **Bypass:** 5.00 dB / 0.838 / 1.50 — the floor both DFN systems beat.
- **Upstream pretrained alone:** strong baseline (16.37 / 0.917 / 3.00) and the
  reference for every paired delta; kept as the comparison anchor, not the
  deployed model, because the fine-tune preserves speech better on this set.
- **Original MetricGAN-style complex U-Net:** research baseline only — weak
  enhancement with regressions vs bypass on impulsive noises; never in the live path.
- **Rejected live variants:** extra +7 dB gain (noisy), bass boost, mixing 3%/6%
  raw noisy input back (gunshots returned), removing EQ (not preferred),
  harsh presence profiles — all rejected by owner listening; the PF03 preset in
  `configs/preset_pf03.json` is the only accepted configuration.