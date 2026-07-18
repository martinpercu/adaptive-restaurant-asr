# Sensitivity analysis — run-20260718T010058Z-0.1.0

## en
- Baseline clean WER: 0.012
- **Top-3 damaging subtypes (NDI):** BB, CA, BC

## es
- Baseline clean WER: 0.365
- **Top-3 damaging subtypes (NDI):** BB, CA, BC

## Findings

Full NDI ranking (identical order for both languages):

| rank | subtype | family | es NDI | en NDI |
|------|---------|--------|--------|--------|
| 1 | BB car construction | B outdoor | 0.633 | 17.14 |
| 2 | CA dining-babble | C front-of-house | 0.534 | 12.55 |
| 3 | BC car-cabin | B outdoor | 0.463 | 10.63 |
| 4 | AC kitchen-babble | A kitchen | 0.433 | 8.59 |
| 5 | BA drive-thru-traffic | B outdoor | 0.331 | 8.30 |
| 6 | AA dishes-cutlery | A kitchen | 0.128 | 3.18 |
| 7 | AB fryer-extractor | A kitchen | 0.092 | 2.69 |

- **Babble outranks stationary, as theory predicts.** The three babble/voice subtypes
  (CA dining, BC car-cabin, AC kitchen) all rank above the two stationary kitchen subtypes
  (AA clatter, AB hum). Overlapping speech is the hardest case — the model cannot separate
  the customer's voice from the background talk. BB (construction) tops the list because it
  is loud broadband energy that masks speech across the whole spectrum.
- **Level monotonicity holds everywhere** (per-level NDI strictly increases 05 → 10 → 15 for
  all 7 subtypes, both languages) — the SNR mixer is behaving correctly. No ceiling effects
  even at level 15 (−5 dB); WER keeps climbing, so the −5 dB choice preserves diagnostic
  discrimination as intended.
- **NDI magnitude differs by language** because it is relative to clean WER. en clean WER is
  ~0.012, so `Δx_rel = (x−x_clean)/max(x_clean, 0.01)` divides by ~0.012 and inflates en NDI
  by ~50×. This is expected: NDI is used for **ranking and relative weighting per language**,
  never for cross-language absolute comparison. The es ranking (clean WER 0.365) is the
  primary steering signal for the es-dominant flywheel work.

## Drive-thru prior check

The primary channel is the drive-thru lane, so family B + BC (car-cabin) should dominate.
The ranking bears this out: **BB (#1)** and **BC (#3)** are family B, and BA (#5) is the third
family-B subtype. CA (#2) is dining-babble — a front-of-house proxy that scores high here
because our BC composites and public babble share acoustic structure; in production the
family-B trio is expected to weigh even more once real drive-thru audio replaces proxies.

## Recommendations

- **Phase-3 (AXIS 1) mitigation targets = top-3: BB, CA, BC.** Prioritize denoising/separation
  chains that help broadband-masking (BB) and overlapping-voice (CA, BC) cases.
- **Phase-4 (AXIS 2) damage-weighted sampling ∝ NDI** — oversample BB/CA/BC/AC noised training
  data hardest; AA/AB get the least weight.
- **Caveat:** the noise bank is built from public proxies (DEMAND kitchen/cafeteria, UrbanSound8K
  outdoor). CB music is deferred (needs MUSAN). Re-run this sensitivity once real field
  drive-thru audio and a `BD wind-weather` subtype exist (phase 7) — the ranking may shift
  family B further up.

## Monotonicity warnings
- none
