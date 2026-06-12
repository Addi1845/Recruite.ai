#!/usr/bin/env python3
"""
Offline pre-computation phase.
Run this ONCE to:
  1. Score all 100K candidates
  2. Upload scores to Supabase (for demo)
  3. Save local parquet for fast ranking

This CAN use APIs, GPU, network — it runs offline.
Runtime: ~10-15 minutes on CPU.
"""

import json, gzip, os, time
import pandas as pd
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

from scoring.skills_scorer import compute_skills_score, compute_assessment_bonus
from scoring.career_scorer import compute_career_score
from scoring.behavioral_scorer import compute_behavioral_score
from scoring.honeypot_detector import is_honeypot
from scoring.bias_mitigator import compute_education_score
from rank import experience_score

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def ingest_all(candidates_path: str, output_parquet: str = "data/features.parquet"):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("Loading candidates...")
    candidates = []
    with open(candidates_path) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    
    print(f"Scoring {len(candidates):,} candidates...")
    rows = []
    batch = []
    BATCH_SIZE = 500
    
    for i, c in enumerate(candidates):
        if i % 5000 == 0:
            print(f"  {i:,}/{len(candidates):,}")
        
        is_trap, _ = is_honeypot(c)
        skills = compute_skills_score(c)
        career = compute_career_score(c)
        behavioral = compute_behavioral_score(c)
        education = compute_education_score(c)
        
        yoe_score = experience_score(c["profile"]["years_of_experience"])
        
        final = (skills * 0.35 + career * 0.25 + yoe_score * 0.15 +
                 behavioral * 0.15 + education + compute_assessment_bonus(c))
        
        row = {
            "candidate_id": c["candidate_id"],
            "final_score": round(min(1.0, max(0.0, final)), 4),
            "skills_score": round(skills, 4),
            "career_score": round(career, 4),
            "behavioral_score": round(behavioral, 4),
            "experience_score": round(yoe_score, 4),
            "title": c["profile"]["current_title"],
            "yoe": c["profile"]["years_of_experience"],
            "location": c["profile"]["location"],
            "country": c["profile"]["country"],
            "is_honeypot": is_trap,
            "last_active_date": c["redrob_signals"]["last_active_date"],
            "open_to_work": c["redrob_signals"]["open_to_work_flag"],
            "notice_period_days": c["redrob_signals"]["notice_period_days"],
        }
        rows.append(row)
        batch.append(row)
        
        # Upload to Supabase in batches
        if len(batch) >= BATCH_SIZE:
            supabase.table("candidate_scores").upsert(batch).execute()
            batch = []
    
    if batch:
        supabase.table("candidate_scores").upsert(batch).execute()
    
    # Save local parquet
    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Saved features.parquet ({len(df):,} rows)")
    
    return df


if __name__ == "__main__":
    df = ingest_all("candidates.jsonl")
    print(df.nlargest(10, "final_score")[["candidate_id", "final_score", "title", "yoe"]])
