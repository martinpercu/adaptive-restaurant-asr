# Mitigation effectiveness — run-20260718T043645Z

| subtype | chain chosen | mean ΔWER rel (worst lang) | mean latency ms |
|---------|--------------|----------------------------|-----------------|
| AA | none | — | — |
| AB | none | — | — |
| AC | none | — | — |
| BA | none | — | — |
| BB | none | — | — |
| BC | none | — | — |
| CA | none | — | — |

## Residual damage (no chain helped → phase-4 priority)
- AA, AB, AC, BA, BB, BC, CA

## Clean-audio harm guard
- en/spectral_gate: clean WER 0.025 → 0.148 (-492.0%)  ⚠ harms clean
- es/spectral_gate: clean WER 0.360 → 0.560 (-55.6%)  ⚠ harms clean
