#!/usr/bin/env python3
"""
Smoke test: load sample candidates, score them, verify basic invariants.
Run from the redrob-ranker/ directory:
    python test_smoke.py
"""

import json
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# == Test 1: Verify all imports work without side effects ==
print("=" * 60)
print("TEST 1: Import verification")
print("=" * 60)

try:
    from scoring.jd_config import CORE_SKILLS, PREFERRED_CITIES, CONSULTING_FIRMS
    from scoring.skills_scorer import compute_skills_score, compute_assessment_bonus
    from scoring.career_scorer import compute_career_score
    from scoring.behavioral_scorer import compute_behavioral_score
    from scoring.honeypot_detector import is_honeypot
    from scoring.bias_mitigator import compute_education_score, apply_bias_checks
    from scoring.reasoning_generator import generate_reasoning
    from rank import experience_score, score_candidate
    print("[PASS] All imports successful -- no side effects triggered")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# == Test 2: Load sample candidates ==
print("\n" + "=" * 60)
print("TEST 2: Load sample candidates")
print("=" * 60)

# Try multiple possible paths for the sample data
sample_paths = [
    os.path.join("..", "[PUB] India_runs_data_and_ai_challenge",
                 "[PUB] India_runs_data_and_ai_challenge",
                 "India_runs_data_and_ai_challenge",
                 "sample_candidates.json"),
    "sample_candidates.json",
    os.path.join("data", "sample_candidates.json"),
]

candidates = None
for path in sample_paths:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"[PASS] Loaded {len(candidates)} candidates from {path}")
        break

if candidates is None:
    print("[FAIL] Could not find sample_candidates.json in any expected location")
    print(f"   Searched: {sample_paths}")
    sys.exit(1)

# Take first 5
test_candidates = candidates[:5]
print(f"   Testing with first {len(test_candidates)} candidates")

# == Test 3: Score each candidate ==
print("\n" + "=" * 60)
print("TEST 3: Score candidates")
print("=" * 60)

all_scores_valid = True

for i, candidate in enumerate(test_candidates):
    cid = candidate["candidate_id"]
    score, breakdown = score_candidate(candidate)
    
    is_hp = breakdown.get("honeypot", False)
    hp_label = "!! HONEYPOT" if is_hp else "CLEAN"
    
    print(f"\n  Candidate {i+1}: {cid}")
    print(f"    Title:     {candidate['profile']['current_title']}")
    print(f"    Company:   {candidate['profile']['current_company']}")
    print(f"    YoE:       {candidate['profile']['years_of_experience']}")
    print(f"    Location:  {candidate['profile']['location']}, {candidate['profile']['country']}")
    print(f"    Status:    {hp_label}")
    print(f"    Final:     {score:.4f}")
    
    if not is_hp:
        print(f"    Breakdown: skills={breakdown['skills']:.3f}, career={breakdown['career']:.3f}, "
              f"exp={breakdown['experience']:.3f}, behavioral={breakdown['behavioral']:.3f}, "
              f"edu={breakdown['education']:.3f}, assess={breakdown['assessment_bonus']:.3f}")
    else:
        print(f"    Reasons:   {breakdown.get('reasons', [])}")
    
    # Generate reasoning
    reasoning = generate_reasoning(candidate, breakdown, rank=i+1)
    print(f"    Reasoning: {reasoning[:120]}...")
    
    # Validate score range
    if not (0.0 <= score <= 1.0):
        print(f"  [FAIL] Score {score} is out of [0.0, 1.0] range!")
        all_scores_valid = False
    
    # Validate breakdown values (if not honeypot)
    if not is_hp:
        for key, val in breakdown.items():
            if key == "final":
                continue
            if not (0.0 <= val <= 1.0):
                print(f"  [FAIL] breakdown['{key}'] = {val} is out of [0.0, 1.0] range!")
                all_scores_valid = False

# == Test 4: Assert all scores valid ==
print("\n" + "=" * 60)
print("TEST 4: Assertions")
print("=" * 60)

assert all_scores_valid, "Some scores were out of [0.0, 1.0] range"
print("[PASS] All scores are within [0.0, 1.0]")

# Verify experience_score edge cases
assert experience_score(6.0) == 1.0, "6 YoE should score 1.0"
assert experience_score(4.5) == 0.85, "4.5 YoE should score 0.85"
assert experience_score(10.0) == 0.85, "10 YoE should score 0.85"
assert experience_score(2.0) == 0.40, "2 YoE should score 0.40"
print("[PASS] experience_score() edge cases pass")

# Verify honeypot detector doesn't crash on minimal profiles
minimal = {
    "candidate_id": "TEST_0000001",
    "profile": {"current_title": "Engineer", "years_of_experience": 5,
                "location": "Delhi", "country": "India",
                "current_company": "TestCo", "current_company_size": "11-50",
                "current_industry": "Tech"},
    "career_history": [],
    "education": [],
    "skills": [],
    "redrob_signals": {}
}
hp_result, hp_reasons = is_honeypot(minimal)
print(f"[PASS] Honeypot detector handles minimal profiles (result: {hp_result})")

# == Test 5: Verify no network imports in rank.py ==
print("\n" + "=" * 60)
print("TEST 5: Network dependency check")
print("=" * 60)

import importlib
rank_module = importlib.import_module("rank")
rank_source = open(rank_module.__file__, "r").read()

forbidden = ["import supabase", "import requests", "import httpx", "import urllib.request"]
violations = [f for f in forbidden if f in rank_source]
if violations:
    print(f"[FAIL] rank.py imports network libraries: {violations}")
    sys.exit(1)
else:
    print("[PASS] rank.py has no network library imports")

# Also check scoring modules
scoring_files = [
    "scoring/jd_config.py",
    "scoring/skills_scorer.py",
    "scoring/career_scorer.py",
    "scoring/behavioral_scorer.py",
    "scoring/honeypot_detector.py",
    "scoring/bias_mitigator.py",
    "scoring/reasoning_generator.py",
]

for sf in scoring_files:
    if os.path.exists(sf):
        src = open(sf, "r").read()
        violations = [f for f in forbidden if f in src]
        if violations:
            print(f"[FAIL] {sf} imports network libraries: {violations}")
            sys.exit(1)

print("[PASS] All scoring modules are network-free")

# == Summary ==
print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
print("\nReady to run on full 100K candidates:")
print("  python rank.py --candidates ./candidates.jsonl --out ./submission.csv")
