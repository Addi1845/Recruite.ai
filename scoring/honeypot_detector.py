# scoring/honeypot_detector.py
from datetime import datetime

def jaccard_similarity(str1: str, str2: str) -> float:
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union if union > 0 else 0.0

def is_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_honeypot: bool, reasons: list[str])
    
    Honeypots have subtly impossible profiles. Don't special-case them —
    just validate logical consistency. A real profile won't fail these.
    """
    reasons = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    
    total_yoe = profile.get("years_of_experience", 0)
    total_months = total_yoe * 12
    REFERENCE_DATE = datetime(2025, 6, 1)
    
    # ── Check 1: Expert in many skills with 0 duration ──────────────────────
    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(zero_duration_experts) >= 5:
        reasons.append(f"expert_zero_duration: {len(zero_duration_experts)} expert skills with 0 months")
    
    # ── Check 2: Skill duration exceeds career length ────────────────────────
    for skill in skills:
        dur = skill.get("duration_months", 0)
        if dur > total_months + 12:  # 12 month buffer
            reasons.append(f"skill_duration_exceeds_career: {skill['name']} ({dur}mo > career {total_months:.0f}mo)")
            break
    
    # ── Check 3: Career start implying impossible YoE ────────────────────────
    career_start_dates = []
    for job in career:
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d")
            career_start_dates.append(start)
        except (ValueError, KeyError, TypeError):
            pass
    
    if career_start_dates:
        earliest_start = min(career_start_dates)
        actual_years = (REFERENCE_DATE - earliest_start).days / 365
        # Allow 3-year gap
        if total_yoe > actual_years + 3.5:
            reasons.append(
                f"impossible_yoe: profile says {total_yoe:.1f}yrs, "
                f"but career starts only {actual_years:.1f}yrs ago"
            )
            
    # ── Check 4: Overlapping employment (same dates, different companies) ────
    dated_jobs = []
    for job in career:
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d")
            end_str = job.get("end_date")
            end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else REFERENCE_DATE
            dated_jobs.append((start, end, job.get("company", ""), job.get("description", "")))
        except (ValueError, KeyError, TypeError):
            pass
    
    for i, (s1, e1, c1, desc1) in enumerate(dated_jobs):
        overlaps = sum(
            1 for j, (s2, e2, c2, desc2) in enumerate(dated_jobs)
            if j != i and s2 < e1 and s1 < e2
        )
        if overlaps >= 3:
            reasons.append(f"impossible_overlapping_jobs: {overlaps+1} concurrent jobs")
            break
            
        # ── Check 6: Copy-Paste Fraud ──────────────────────────────────────────
        for j, (s2, e2, c2, desc2) in enumerate(dated_jobs):
            if j > i and len(desc1) > 50 and len(desc2) > 50:
                sim = jaccard_similarity(desc1, desc2)
                if sim > 0.85:
                    reasons.append(f"copy_paste_fraud: high description overlap between {c1} and {c2}")
                    break
    
    # ── Check 5: Negative duration in a job ─────────────────────────────────
    for job in career:
        dur = job.get("duration_months", 0)
        if dur < 0:
            reasons.append(f"negative_job_duration: {job.get('company', '?')} has {dur}mo")
            
    # ── Check 7: Future skills in past jobs ──────────────────────────────────
    SKILL_RELEASE_YEARS = {
        "gpt-4": 2023,
        "chatgpt": 2022,
        "llama": 2023,
        "mistral": 2023,
        "claude": 2023,
        "pytorch": 2016,
        "tensorflow": 2015,
        "bert": 2018,
        "transformers": 2017
    }
    
    for job in career:
        desc = job.get("description", "").lower()
        try:
            end_str = job.get("end_date")
            end_year = datetime.strptime(end_str, "%Y-%m-%d").year if end_str else REFERENCE_DATE.year
            for skill, year in SKILL_RELEASE_YEARS.items():
                if skill in desc and end_year < year:
                    reasons.append(f"impossible_skill_timeline: used {skill} before {year} in job ending {end_year}")
                    break
        except (ValueError, KeyError, TypeError):
            pass

    # ── Check 8: Career start < Latest Education End ─────────────────────────
    latest_edu_end = None
    for edu in education:
        try:
            end_str = edu.get("end_date")
            if end_str:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                if not latest_edu_end or end_date > latest_edu_end:
                    latest_edu_end = end_date
        except (ValueError, KeyError, TypeError):
            pass
            
    if career_start_dates and latest_edu_end:
        earliest_start = min(career_start_dates)
        if earliest_start < latest_edu_end:
            # Let's be lenient: if they started career way before education ended, it could be a master's or part-time.
            # Only flag if it's a huge discrepancy. Or, per the report, just flag it. Let's add a small buffer or just flag.
            reasons.append(f"career_before_education: career started {earliest_start.year} before education ended {latest_edu_end.year}")

    return len(reasons) > 0, reasons
