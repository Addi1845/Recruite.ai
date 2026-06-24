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

from rank import score_candidate

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def ingest_all(candidates_path: str, output_parquet: str = "data/features.parquet"):
    # Initialize Supabase client only if keys are present
    supabase = None
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("Loading candidates...")
    candidates = []
    with open(candidates_path, encoding='utf-8') as f:
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
        
        final_score, breakdown = score_candidate(c)
        
        row = {
            "candidate_id": c["candidate_id"],
            "final_score": round(final_score, 4),
            "skills_score": breakdown.get("skills", 0.0),
            "career_score": breakdown.get("career", 0.0),
            "behavioral_score": breakdown.get("behavioral", 0.0),
            "experience_score": breakdown.get("experience", 0.0),
            "title": c["profile"]["current_title"],
            "yoe": c["profile"]["years_of_experience"],
            "location": c["profile"]["location"],
            "country": c["profile"]["country"],
            "is_honeypot": breakdown.get("honeypot", False),
            "last_active_date": c.get("redrob_signals", {}).get("last_active_date", ""),
            "open_to_work": c.get("redrob_signals", {}).get("open_to_work_flag", False),
            "notice_period_days": c.get("redrob_signals", {}).get("notice_period_days", 60),
        }
        rows.append(row)
        batch.append(row)
        
        # Upload to Supabase in batches
        if supabase and len(batch) >= BATCH_SIZE:
            try:
                supabase.table("candidate_scores").upsert(batch).execute()
            except Exception as e:
                print(f"Supabase upsert failed: {e}")
            batch = []
    
    if supabase and batch:
        try:
            supabase.table("candidate_scores").upsert(batch).execute()
        except Exception as e:
            print(f"Supabase upsert failed: {e}")
    
    # Save local parquet
    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Saved features.parquet ({len(df):,} rows)")
    
    return df


if __name__ == "__main__":
    df = ingest_all("..\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl")
    print(df.nlargest(10, "final_score")[["candidate_id", "final_score", "title", "yoe"]])
