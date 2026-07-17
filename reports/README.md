# reports/

Generated evaluation artifacts (JSON + PNG): sensitivity matrices, NDI rankings,
judge calibration, per-phase metric tables. **Gitignored** except this README.

Layout (populated by later phases):

```
reports/
├── sensitivity/<run_id>/   # matrix.parquet, ndi.json, heatmap-{lang}.png   (phase 2)
├── judge/                  # calibration reports                            (phase 6)
└── baseline/               # phase-1 baseline WER reports
```

Never hand-edit these files (CLAUDE.md hard rule 3).
