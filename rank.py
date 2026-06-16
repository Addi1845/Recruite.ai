#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker — v2 (Enhanced)
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: <=5 min, <=16 GB RAM, CPU only, no network calls.

v2 enhancements over v1:
  - TF-IDF cosine similarity with JD text (captures semantic overlap)
  - Skill credibility scoring (endorsements vs duration vs proficiency)
  - AI/ML specialization score (specialist vs generalist)
  - Career trajectory analysis (upward mobility + ML pivot detection)
  - Salary budget alignment
  - Re-optimized weight distribution for NDCG@10
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

# ── Original scoring modules ────────────────────────────────────────────────
from scoring.jd_config import CORE_SKILLS
from scoring.skills_scorer import compute_skills_score, compute_assessment_bonus
from scoring.career_scorer import compute_career_score
from scoring.behavioral_scorer import compute_behavioral_score
from scoring.honeypot_detector import is_honeypot
from scoring.bias_mitigator import compute_education_score, apply_bias_checks
from scoring.reasoning_generator import generate_reasoning

# ── NEW v2 scoring modules ──────────────────────────────────────────────────
from scoring.tfidf_scorer import compute_tfidf_score
from scoring.advanced_features import (
    compute_skill_credibility,
    compute_specialization_score,
    compute_career_trajectory,
    compute_salary_fit,
)


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
    
    v2 weight distribution (optimized for NDCG@10):
    ─────────────────────────────────────────────────────
    Signal              Weight    Rationale
    ─────────────────────────────────────────────────────
    skills              0.28      Core JD match (slightly reduced to make room)
    tfidf               0.10      Semantic text overlap with JD
    career              0.18      Career quality + production ML evidence
    experience          0.10      Years of experience fit
    behavioral          0.10      Availability + engagement signals
    specialization      0.07      AI/ML specialist vs generalist
    credibility         0.05      Skill claim believability
    trajectory          0.04      Career progression toward AI
    salary_fit          0.03      Budget alignment
    education           max 0.03  Tier bonus (compressed)
    assessment          max 0.05  Redrob platform test scores
    ─────────────────────────────────────────────────────
    Total               ~1.03     (clamped to 1.0)
    """
    # ── Honeypot detection ───────────────────────────────────────────────────
    is_trap, trap_reasons = is_honeypot(candidate)
    if is_trap:
        return 0.0, {"honeypot": True, "reasons": trap_reasons}
    
    # ── Original component scores ────────────────────────────────────────────
    skills = compute_skills_score(candidate)
    career = compute_career_score(candidate)
    yoe = experience_score(candidate["profile"]["years_of_experience"])
    behavioral = compute_behavioral_score(candidate)
    education = compute_education_score(candidate)
    assessment_bonus = compute_assessment_bonus(candidate)
    
    # ── NEW v2 component scores ──────────────────────────────────────────────
    tfidf = compute_tfidf_score(candidate)
    credibility = compute_skill_credibility(candidate)
    specialization = compute_specialization_score(candidate)
    trajectory = compute_career_trajectory(candidate)
    salary = compute_salary_fit(candidate)
    
    # ── Weighted combination (v2 optimized) ──────────────────────────────────
    # Education and assessment are already pre-scaled (max 0.03 and 0.05)
    education_scaled = min(0.03, education * 0.6)  # compress from max 0.05 to 0.03
    assessment_scaled = min(0.05, assessment_bonus * 0.625)  # compress from max 0.08 to 0.05
    
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
        "skills": round(skills, 3),
        "tfidf": round(tfidf, 3),
        "career": round(career, 3),
        "experience": round(yoe, 3),
        "behavioral": round(behavioral, 3),
        "specialization": round(specialization, 3),
        "credibility": round(credibility, 3),
        "trajectory": round(trajectory, 3),
        "salary_fit": round(salary, 3),
        "education": round(education_scaled, 3),
        "assessment_bonus": round(assessment_scaled, 3),
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
    print(f"[{time.strftime('%H:%M:%S')}] Starting Redrob AI Ranker v2 (Enhanced)...")
    
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
    print(f"[{time.strftime('%H:%M:%S')}] Scoring candidates (v2 enhanced pipeline)...")
    
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
