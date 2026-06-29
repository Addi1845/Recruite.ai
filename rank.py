#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker — v3 (Speed-Optimized)
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: <=5 min, <=16 GB RAM, CPU only, no network calls.

v3 speed optimizations:
  - Multiprocessing: parallel scoring across all CPU cores
  - Chunked batch processing to minimize IPC overhead
  - Pre-compiled regex patterns cached at module level
  - Fast-path early exit in honeypot detector
  - Optimized Jaccard: pre-split sets, short-circuit on length mismatch
  - TF-IDF bigrams capped at a reasonable token limit per candidate
"""

import argparse
import csv
import json
import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

# ── Original scoring modules ────────────────────────────────────────────────
from scoring.jd_config import CORE_SKILLS
from scoring.skills_scorer import compute_skills_score, compute_assessment_bonus
from scoring.career_scorer import compute_career_score
from scoring.behavioral_scorer import compute_behavioral_score
from scoring.honeypot_detector import is_honeypot
from scoring.bias_mitigator import compute_education_score, apply_bias_checks
from scoring.reasoning_generator import generate_reasoning

# ── v2/v3 scoring modules ───────────────────────────────────────────────────
from scoring.tfidf_scorer import compute_tfidf_score
from scoring.advanced_features import (
    compute_skill_credibility,
    compute_specialization_score,
    compute_career_trajectory,
    compute_salary_fit,
)


def experience_score(yoe: float) -> float:
    """Score experience years fit. JD says 5-9 preferred, 4-10 acceptable."""
    if 5.0 <= yoe <= 9.0:
        return 1.0
    elif 4.0 <= yoe < 5.0:
        return 0.85
    elif 9.0 < yoe <= 10.0:
        return 0.85
    elif 10.0 < yoe <= 12.0:
        return 0.70
    elif 3.0 <= yoe < 4.0:
        return 0.55
    elif 12.0 < yoe <= 15.0:
        return 0.55
    elif 2.0 <= yoe < 3.0:
        return 0.40
    else:
        return 0.30


def score_candidate(candidate: dict) -> tuple[float, dict]:
    """
    Returns (final_score, score_breakdown).

    v3 weight distribution (optimized for NDCG@10):
    ─────────────────────────────────────────────────────
    Signal              Weight    Rationale
    ─────────────────────────────────────────────────────
    skills              0.28      Core JD match
    tfidf               0.10      Semantic text overlap with JD
    career              0.18      Career quality + production ML evidence
    experience          0.10      Years of experience fit
    behavioral          0.10      Availability + engagement signals
    specialization      0.07      AI/ML specialist vs generalist
    credibility         0.05      Skill claim believability
    trajectory          0.04      Career progression toward AI
    salary_fit          0.03      Budget alignment
    education           max 0.05  Tier bonus
    assessment          max 0.08  Redrob platform test scores
    ─────────────────────────────────────────────────────
    Total               ~1.08     (clamped to 1.0)
    """
    # ── Honeypot detection (fast-path: if caught, skip expensive scoring) ─────
    is_trap, trap_reasons = is_honeypot(candidate)
    if is_trap:
        return 0.0, {"honeypot": True, "reasons": trap_reasons}

    # ── Component scores ──────────────────────────────────────────────────────
    skills          = compute_skills_score(candidate)
    career          = compute_career_score(candidate)
    yoe             = experience_score(candidate["profile"]["years_of_experience"])
    behavioral      = compute_behavioral_score(candidate)
    education       = compute_education_score(candidate)
    assessment_bonus = compute_assessment_bonus(candidate)

    # v3 signals
    tfidf           = compute_tfidf_score(candidate)
    credibility     = compute_skill_credibility(candidate)
    specialization  = compute_specialization_score(candidate)
    trajectory      = compute_career_trajectory(candidate)
    salary          = compute_salary_fit(candidate)

    education_scaled  = min(0.05, education)
    assessment_scaled = min(0.08, assessment_bonus)

    raw = (
        skills         * 0.28 +
        tfidf          * 0.10 +
        career         * 0.18 +
        yoe            * 0.10 +
        behavioral     * 0.10 +
        specialization * 0.07 +
        credibility    * 0.05 +
        trajectory     * 0.04 +
        salary         * 0.03 +
        education_scaled +
        assessment_scaled
    )

    final = apply_bias_checks(candidate, raw)

    breakdown = {
        "skills":           round(skills, 3),
        "tfidf":            round(tfidf, 3),
        "career":           round(career, 3),
        "experience":       round(yoe, 3),
        "behavioral":       round(behavioral, 3),
        "specialization":   round(specialization, 3),
        "credibility":      round(credibility, 3),
        "trajectory":       round(trajectory, 3),
        "salary_fit":       round(salary, 3),
        "education":        round(education_scaled, 3),
        "assessment_bonus": round(assessment_scaled, 3),
        "final":            round(final, 3),
    }

    return min(1.0, max(0.0, final)), breakdown


def _fast_prefilter(candidate: dict) -> bool:
    """Ultra-fast pre-filter: return True if candidate is worth scoring.
    
    Eliminates obvious non-candidates in microseconds before the expensive
    scoring pipeline runs. Only rejects candidates with zero signal.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    
    # Must have at least some skills listed
    if not skills:
        return False
    
    # Must have a title (basic completeness)
    title = profile.get("current_title", "")
    if not title:
        return False
    
    # If title is clearly irrelevant AND no ML-adjacent skills, skip
    BAD_DOMAINS = {"accountant", "lawyer", "attorney", "chef", "nurse", "doctor",
                   "teacher", "professor", "hr manager", "civil engineer",
                   "mechanical engineer", "electrical engineer", "sales manager",
                   "marketing manager", "content writer", "graphic designer"}
    title_lower = title.lower()
    if any(b in title_lower for b in BAD_DOMAINS):
        skill_names = " ".join(s.get("name", "").lower() for s in skills)
        if "python" not in skill_names and "ml" not in skill_names and "ai" not in skill_names:
            return False
    
    return True


def _score_batch(batch: list[dict]) -> list[dict]:
    """Score a batch of candidates. Runs in a worker process."""
    results = []
    for candidate in batch:
        try:
            # Fast pre-filter: skip obvious non-candidates instantly
            if not _fast_prefilter(candidate):
                results.append({
                    "candidate_id": candidate["candidate_id"],
                    "score": 0.0,
                    "breakdown": {"honeypot": True, "reasons": ["prefilter_reject"]},
                    "candidate": candidate,
                })
                continue
            score, breakdown = score_candidate(candidate)
        except Exception as e:
            score, breakdown = 0.0, {"honeypot": True, "reasons": [f"error: {e}"]}
        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": score,
            "breakdown": breakdown,
            "candidate": candidate,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (0 = auto-detect CPU count)")
    args = parser.parse_args()

    start_time = time.time()

    # Auto-detect CPU cores, leave 1 free for the OS
    n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)
    print(f"[{time.strftime('%H:%M:%S')}] Starting Redrob AI Ranker v3 (Speed-Optimized)...")
    print(f"[{time.strftime('%H:%M:%S')}] Using {n_workers} worker processes on {mp.cpu_count()} CPUs")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found")
        sys.exit(1)

    # ── Load candidates ───────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Loading candidates from {candidates_path}...")
    candidates = []

    if candidates_path.suffix == ".gz":
        import gzip
        with gzip.open(candidates_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    total = len(candidates)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {total:,} candidates")

    # ── Parallel scoring ──────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Scoring candidates (parallel pipeline)...")

    # Shuffle to spread hard candidates evenly, then split into exactly N_WORKERS
    # chunks — one per process. This eliminates straggler skew AND IPC overhead
    # (only n_workers pickle round-trips instead of 500).
    import random
    random.seed(42)  # Deterministic shuffle
    random.shuffle(candidates)
    chunks = [candidates[i::n_workers] for i in range(n_workers)]  # round-robin split

    all_results = []
    honeypot_count = 0

    with mp.Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_score_batch, chunks)  # one result per worker
        for batch_results in chunk_results:
            all_results.extend(batch_results)
            honeypot_count += sum(1 for r in batch_results if r["breakdown"].get("honeypot"))

    elapsed = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Scoring complete in {elapsed:.1f}s")
    print(f"  Honeypots detected: {honeypot_count:,}")

    # ── Sort and take top N ───────────────────────────────────────────────────
    all_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_n = all_results[:args.top_n]

    # ── Generate reasoning & write CSV ────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Generating reasoning for top {args.top_n}...")
    out_path = Path(args.out)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, result in enumerate(top_n, start=1):
            reasoning = generate_reasoning(result["candidate"], result["breakdown"], rank)
            writer.writerow([result["candidate_id"], rank, f"{result['score']:.4f}", reasoning])

    total_time = time.time() - start_time
    print(f"\nDone! Output written to: {out_path}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Top score:      {top_n[0]['score']:.4f} ({top_n[0]['candidate_id']})")
    print(f"   Rank 100 score: {top_n[-1]['score']:.4f} ({top_n[-1]['candidate_id']})")


if __name__ == "__main__":
    # Required for Windows multiprocessing (spawn-based)
    mp.freeze_support()
    main()
