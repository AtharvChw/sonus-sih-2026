# v18 paired validation review

Verified 480 identical mixtures per system; package and export checksums match.

| System | Output SNR dB | SI-SNR dB | STOI | PESQ-WB |
|---|---:|---:|---:|---:|
| bypass | 5.000 | 5.050 | 0.8379 | 1.4986 |
| pretrained | 16.370 | 16.502 | 0.9166 | 3.0025 |
| impulse_weighted | 17.008 | 17.154 | 0.9174 | 3.0129 |
| no_oversampling | 16.987 | 17.108 | 0.9183 | 3.0274 |

## Paired losses and gains

### impulse_weighted

Material losses versus pretrained: {'stoi_drop_over_0.02': 20, 'pesq_drop_over_0.1': 95, 'snr_drop_over_1db': 2}.

- snr_db: mean delta 0.63791; exploratory cluster CI [0.43919316157909793, 0.8708759834821953].
- si_snr_db: mean delta 0.65182; exploratory cluster CI [0.4107603393377608, 0.9697006591104448].
- stoi: mean delta 0.00083; exploratory cluster CI [-0.005900678357552907, 0.005320218878287931].
- pesq_wb: mean delta 0.01040; exploratory cluster CI [-0.03360172269865871, 0.06619901105140646].
### no_oversampling

Material losses versus pretrained: {'stoi_drop_over_0.02': 12, 'pesq_drop_over_0.1': 92, 'snr_drop_over_1db': 3}.

- snr_db: mean delta 0.61726; exploratory cluster CI [0.44753290539761403, 0.8227977920654553].
- si_snr_db: mean delta 0.60518; exploratory cluster CI [0.4145951876489586, 0.8595369068527994].
- stoi: mean delta 0.00174; exploratory cluster CI [-0.004354379330717814, 0.006034676338857254].
- pesq_wb: mean delta 0.02491; exploratory cluster CI [-0.022235893520216148, 0.08436395764350889].

These are validation means, not per-condition guarantees. Three speakers are insufficient for final generalization claims.
ONNX tensor parity passed; Rust waveform parity and Pi runtime remain unverified.
No deployment selection made by this script.
