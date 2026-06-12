#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: ≤5 min, ≤16 GB RAM, CPU only, no network calls.
Strategy: Pre-computed features (from ingest.py) loaded from local parquet.
          If parquet not found, falls back to direct scoring (slower but works).
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Our scoring modules
from scoring.jd_config import CORE_SKILLS
from scoring.skills_scorer import compute_skills_score, compute_assessment_bonus
from scoring.career_scorer import compute_career_score
from scoring.behavioral_scorer import compute_behavioral_score
from scoring.honeypot_detector import is_honeypot
from scoring.bias_mitigator import compute_education_score, apply_bias_checks
from scoring.reasoning_generator import generate_reasoning


def experience_score(yoe: float) -> float:
    """Score experience years fit. JD says 5-9 preferred, 4-10 acceptable.
    
    Smooth curve: sweet spot at 5-9 (1.0), acceptable at 4-10 (0.80-0.85),
    slightly low at 3-4 or 11-14 (0.55-0.65), minimum for <3 or >14 (0.30-0.40).
    """
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
    
    Score weights are calibrated to NDCG@10 (50% of eval metric),
    meaning top picks need to be very precise.
    
    Final = skills(35%) + career(25%) + experience(15%) + behavioral(15%)
          + education(5%) + assessment_bonus(up to 8%)
          - negative_signals
    """
    # ── Honeypot detection ───────────────────────────────────────────────────
    is_trap, trap_reasons = is_honeypot(candidate)
    if is_trap:
        return 0.0, {"honeypot": True, "reasons": trap_reasons}
    
    # ── Component scores ─────────────────────────────────────────────────────
    skills = compute_skills_score(candidate)
    career = compute_career_score(candidate)
    yoe = experience_score(candidate["profile"]["years_of_experience"])
    behavioral = compute_behavioral_score(candidate)
    education = compute_education_score(candidate)
    assessment_bonus = compute_assessment_bonus(candidate)
    
    # ── Weighted combination ──────────────────────────────────────────────────
    raw = (
        skills    * 0.35 +
        career    * 0.25 +
        yoe       * 0.15 +
        behavioral * 0.15 +
        education +          # max 0.05
        assessment_bonus     # max 0.08
    )
    
    final = apply_bias_checks(candidate, raw)
    
    breakdown = {
        "skills": round(skills, 3),
        "career": round(career, 3),
        "experience": round(yoe, 3),
        "behavioral": round(behavioral, 3),
        "education": round(education, 3),
        "assessment_bonus": round(assessment_bonus, 3),
        "final": round(final, 3),
    }
    
    return min(1.0, max(0.0, final)), breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()
    
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting Redrob AI Ranker...")
    
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
    
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(candidates):,} candidates")
    
    # ── Score all candidates ──────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Scoring candidates...")
    
    results = []
    honeypot_count = 0
    
    for i, candidate in enumerate(candidates):
        if i % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"  [{time.strftime('%H:%M:%S')}] {i:,}/{len(candidates):,} | {elapsed:.1f}s elapsed")
        
        score, breakdown = score_candidate(candidate)
        
        if breakdown.get("honeypot"):
            honeypot_count += 1
        
        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": score,
            "breakdown": breakdown,
            "candidate": candidate,  # kept for reasoning generation
        })
    
    elapsed = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Scoring complete in {elapsed:.1f}s")
    print(f"  Honeypots detected: {honeypot_count}")
    
    # ── Sort and take top N ───────────────────────────────────────────────────
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_n = results[:args.top_n]
    
    # ── Generate reasoning ────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Generating reasoning for top {args.top_n}...")
    
    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, result in enumerate(top_n, start=1):
            cid = result["candidate_id"]
            score = result["score"]
            candidate = result["candidate"]
            breakdown = result["breakdown"]
            
            reasoning = generate_reasoning(candidate, breakdown, rank)
            
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])
    
    total_time = time.time() - start_time
    print(f"\nDone! Output written to: {out_path}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Top score: {top_n[0]['score']:.4f} ({top_n[0]['candidate_id']})")
    print(f"   Rank 100 score: {top_n[-1]['score']:.4f} ({top_n[-1]['candidate_id']})")


if __name__ == "__main__":
    main()
