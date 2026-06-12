# Redrob AI Candidate Ranker

Multi-signal, bias-aware candidate ranking engine for the India Runs Data & AI Challenge — scores 100K profiles and returns the top 100 most hirable candidates for a Senior AI Engineer JD, ranked best-fit first.

## Setup

```bash
pip install -r requirements.txt
```

## Pre-computation (one-time, internet OK)

```bash
python ingest.py
```

This scores all 100K candidates and uploads to Supabase for the demo sandbox. Also saves `data/features.parquet` locally.

## Ranking (≤5 min, CPU only, no network)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Validate

```bash
python validate_submission.py submission.csv
```

## Demo Sandbox

```bash
cd demo && streamlit run app.py
```

## Architecture

Multi-signal, bias-aware scoring pipeline with five components:

- **Skills Scorer** — Coherence-checked skill matching: JD-weighted skill lookup cross-validated against career description evidence. Penalizes keyword stuffers (e.g. HR Manager listing "embeddings" with no career backing).
- **Career Scorer** — Semantic analysis of career history: detects production ML deployment signals, penalizes pure-consulting or research-only backgrounds, rewards tenure in relevant roles.
- **Behavioral Scorer** — Converts Redrob platform engagement signals (last active, response rate, notice period, GitHub activity) into an availability/readiness multiplier.
- **Honeypot Detector** — Logical consistency validation: catches impossible timelines, zero-duration expert claims, overlapping employment, and negative job durations.
- **Bias Mitigator** — Compresses education/institution tier to ≤5% of final score. No penalty for career gaps, salary expectations, or profile completeness.

## Bias Mitigation

- **Institution prestige**: Education contributes max 5% of final score
- **Company prestige**: FAANG ≠ auto-high score; measures what was built, not where
- **Age/YoE**: Smooth scoring curve — no hard cutoffs (4 yrs = 0.85, 12 yrs = 0.65)
- **Gender**: Names are anonymized; zero name-based signals used
- **Location**: Scoring-based, not hard filter; willing to relocate = partial credit
- **Salary expectations**: Not used as a penalty signal
- **Profile completeness**: Minimal weight (0.03 max effect)
- **Career gaps**: No penalty; analyzes career quality, not continuity

## Compute Environment

- Runtime: ~3 min on 16GB CPU for 100K candidates
- No GPU required for ranking
- No network calls during ranking step
- Python 3.11+
