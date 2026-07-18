# Field Recording Protocol

How to capture real restaurant audio so it faithfully replaces the public/synthetic
bootstrap. **Reviewed by the product owner before the first real-store deployment.**

Public-noise NDI rankings do not survive contact with real stores — real mics change gain
structure, frequency response, and room acoustics. Re-run the sensitivity analysis after each
batch of real audio; expect the ranking to shift.

## Noise-only captures (per store, per daypart)

For each daypart — **open**, **rush**, **close** — capture **10 minutes of noise only** (no
customer speech) at the actual mic position:

1. **Drive-thru mic first** — it is the primary channel.
2. Then the counter mic.

At outdoor mics, **log weather/wind** per capture. Once real recordings exist, register a
dedicated `BD wind-weather` subtype (reserved in the taxonomy; kept out of the bootstrap grid
because no public proxy is faithful) and curate these clips into the noise bank as
`own-recording` under the correct subtype. They progressively replace the public proxies.

## Real order audio (ground truth)

- **Consent:** post the legally-required call-recording signage; confirm the lawful basis for
  your jurisdiction (placeholder — legal review required per deployment).
- **Gain staging:** target peak −12 dBFS; disable AGC if avoidable. Record the mic model and
  gain in the sidecar `meta` of every capture.
- **Format:** 16 kHz mono PCM at ingest (convert, don't resample in the engine).
- **Volume:** ≥ 500 real utterances, transcribed via the review queue
  (`python -m ars.flywheel.review`) → the first real ground-truth eval set
  `eval-real-<lang>-v1`.

## After real data exists

- All promotion gates run on the **real** eval sets; synthetic sets stay as regression suites.
- A US deployment must include an **`es-mx` eval slice** before its gates count (Mexican /
  Caribbean accents dominate US Spanish; the bootstrap has no no-auth es-mx read corpus).
- Curate MUSAN + UrbanSound8K + Common Voice enrichment as they become available (each behind
  a DECISIONS.md entry).

## Sidecar metadata (per capture)

```json
{"store_id": "...", "mic": "...", "gain_db": 0, "daypart": "rush",
 "weather": "windy 20km/h", "captured_at": "...Z", "channel": "drive-thru"}
```
