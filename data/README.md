# data/

All contents are **gitignored** except manifests (`manifest.parquet`, `dataset.json`),
license sidecars, and READMEs (CLAUDE.md hard rule 2).

Layout (plan/01-conventions.md §5):

```
data/
├── clean/<lang>/              # downloaded clean speech + manifest.parquet   (phase 0)
├── datasets/<dataset_id>/     # TTS corpus, mixed eval matrices             (phase 0/2)
├── noise_bank/<SUBTYPE>/      # curated noise clips + manifest.parquet      (phase 2)
├── fixtures/<lang>/           # tiny WER fixtures (make fixtures)           (phase 0)
├── _staging/noise_sources/    # raw downloaded noise, pre-curation         (phase 0)
├── telemetry/YYYY-MM-DD.jsonl # request telemetry                          (phase 1)
└── db/ars.db                  # SQLite operational store (WAL)             (phase 1)
```

Bootstrap with `make download-data` (public datasets) and `make tts-corpus` (Piper TTS).
