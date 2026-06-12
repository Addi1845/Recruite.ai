# scoring/honeypot_detector.py
from datetime import datetime

def is_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_honeypot: bool, reasons: list[str])
    
    Honeypots have subtly impossible profiles. Don't special-case them —
    just validate logical consistency. A real profile won't fail these.
    """
    reasons = []
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    total_yoe = profile["years_of_experience"]
    total_months = total_yoe * 12
    
    # ── Check 1: Expert in many skills with 0 duration ──────────────────────
    # Legitimate: someone can claim expert with minimal duration (1-3 months).
    # Impossible: 8+ skills at expert level with 0 months each.
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
    # Profile says 8 YoE but earliest job started 3 years ago → honeypot
    career_start_dates = []
    for job in career:
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d")
            career_start_dates.append(start)
        except (ValueError, KeyError):
            pass
    
    if career_start_dates:
        earliest_start = min(career_start_dates)
        actual_years = (datetime.now() - earliest_start).days / 365
        # Allow 3-year gap (career breaks, pre-tracked roles, etc.)
        if total_yoe > actual_years + 3.5:
            reasons.append(
                f"impossible_yoe: profile says {total_yoe:.1f}yrs, "
                f"but career starts only {actual_years:.1f}yrs ago"
            )
    
    # ── Check 4: Overlapping employment (same dates, different companies) ────
    # Sort by start date and check for impossible overlaps
    dated_jobs = []
    for job in career:
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d")
            end_str = job.get("end_date")
            end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else datetime.now()
            dated_jobs.append((start, end, job.get("company", "")))
        except (ValueError, KeyError):
            pass
    
    # Check for 3+ simultaneously active jobs (2 is fine — people moonlight)
    for i, (s1, e1, c1) in enumerate(dated_jobs):
        overlaps = sum(
            1 for j, (s2, e2, c2) in enumerate(dated_jobs)
            if j != i and s2 < e1 and s1 < e2
        )
        if overlaps >= 3:
            reasons.append(f"impossible_overlapping_jobs: {overlaps+1} concurrent jobs")
            break
    
    # ── Check 5: Negative duration in a job ─────────────────────────────────
    for job in career:
        dur = job.get("duration_months", 0)
        if dur < 0:
            reasons.append(f"negative_job_duration: {job.get('company', '?')} has {dur}mo")
    
    return len(reasons) > 0, reasons
