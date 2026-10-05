# Decisions (short ablation)

## Selected: `no_oversampling` (six-epoch DeepFilterNet3 fine-tune, epoch-6 best)

480-mixture development validation, raw outputs, identical mixtures per system:

| System | Output SNR (dB) | SI-SNR (dB) | STOI | PESQ-WB |
|---|---|---:|---:|---:|
| Bypass (no enhancement) | 5.000 | 5.0501 | 0.837897 | 1.498556 |
| Upstream pretrained DFN3 | 16.369689 | 16.502343 | 0.916571 | 3.002469 |
| `impulse_weighted` (SESA train x2.0) | 17.007600 | 17.154162 | 0.917401 | 3.012865 |
| `no_oversampling` (SESA train x1.0, selected) | 16.986951 | 17.107518 | 0.918306 | 3.027380 |

## Why `no_oversampling` over `impulse_weighted`

- Highest STOI (0.918306) and PESQ-WB (3.027380) of the comparison.
- Fewest material speech regressions vs pretrained: STOI drop > 0.02 in 12/480
  (vs 20/480 weighted) and PESQ drop > 0.1 in 92/480 (vs 95/480 weighted);
  SNR drop > 1 dB in 3/480 (vs 2/480 weighted).
- SNR essentially tied (+0.6173 dB vs +0.6379 dB mean delta over pretrained,
  overlapping exploratory CIs).

The choice is a speech-preservation tradeoff for runtime testing, not a claim of
universal superiority: STOI/PESQ gain CIs cross zero, and 92/480 PESQ regressions
remain under review. Higher training loss (0.6613889 vs 0.6610152) did not predict
the perceptual tradeoff; selection used validation metrics, never the sealed test.

## Why not the alternatives

- **Bypass:** 5.0 dB / 0.8379 / 1.4986 — the floor both DFN systems beat.
- **Upstream pretrained alone:** strong baseline (16.370 / 0.9166 / 3.0025) and the
  reference for every paired delta; kept as the comparison anchor, not the
  deployed model, because the fine-tune preserves speech better on this set.
- **Original MetricGAN-style complex U-Net:** research baseline only — weak
  enhancement with regressions vs bypass on impulsive noises; never in the live path.
- **Rejected live variants:** extra +7 dB gain (noisy), bass boost, mixing 3%/6%
  raw noisy input back (gunshots returned), removing EQ (not preferred),
  harsh presence profiles — all rejected by owner listening; the PF03 preset in
  `configs/preset_pf03.json` is the only accepted configuration.