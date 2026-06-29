# scoring/honeypot_detector.py
from datetime import datetime
from functools import lru_cache

# ── Module-level constants (created ONCE, not per-candidate call) ──────────
_REFERENCE_DATE = datetime(2025, 6, 1)
_REFERENCE_YEAR = 2025

_SKILL_RELEASE_YEARS = {
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


@lru_cache(maxsize=8192)
def _parse_date_cached(date_str: str):
    """Parse date string once, cache result. A date like '2020-06-01' appears
    in thousands of candidates — this saves ~600k strptime calls on 100k candidates."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=8192)
def _get_end_year(date_str: str) -> int:
    """Return year from date string, cached."""
    dt = _parse_date_cached(date_str)
    return dt.year if dt else _REFERENCE_YEAR


def jaccard_similarity(set1: set, set2: set) -> float:
    """Fast Jaccard on pre-built sets with early-exit on length mismatch."""
    if not set1 or not set2:
        return 0.0
    # Fast early-exit: if one set is 5x larger, similarity can't exceed ~0.2
    ratio = max(len(set1), len(set2)) / max(1, min(len(set1), len(set2)))
    if ratio > 5:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1) + len(set2) - intersection
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
        if dur > total_months + 12:
            reasons.append(f"skill_duration_exceeds_career: {skill['name']} ({dur}mo > career {total_months:.0f}mo)")
            break

    # ── Check 3: Career start implying impossible YoE ────────────────────────
    career_start_dates = []
    for job in career:
        sd = job.get("start_date")
        if sd:
            dt = _parse_date_cached(sd)
            if dt:
                career_start_dates.append(dt)

    if career_start_dates:
        earliest_start = min(career_start_dates)
        actual_years = (_REFERENCE_DATE - earliest_start).days / 365
        if total_yoe > actual_years + 3.5:
            reasons.append(
                f"impossible_yoe: profile says {total_yoe:.1f}yrs, "
                f"but career starts only {actual_years:.1f}yrs ago"
            )

    # ── Check 4 & 6: Overlapping employment + Copy-Paste Fraud ──────────────
    dated_jobs = []
    for job in career:
        sd = job.get("start_date")
        ed = job.get("end_date")
        start = _parse_date_cached(sd) if sd else None
        end = _parse_date_cached(ed) if ed else _REFERENCE_DATE
        if start:
            dated_jobs.append((start, end, job.get("company", ""), job.get("description", "")))

    for i, (s1, e1, c1, desc1) in enumerate(dated_jobs):
        overlaps = sum(
            1 for j, (s2, e2, c2, desc2) in enumerate(dated_jobs)
            if j != i and s2 < e1 and s1 < e2
        )
        if overlaps >= 3:
            reasons.append(f"impossible_overlapping_jobs: {overlaps+1} concurrent jobs")
            break

    # Copy-paste fraud: pre-tokenize once, check pairs
    tokenized_descs = [
        (i, c, set(desc.lower().split()))
        for i, (s, e, c, desc) in enumerate(dated_jobs)
        if len(desc) > 50
    ]
    found_fraud = False
    for idx_a in range(len(tokenized_descs)):
        if found_fraud:
            break
        i, c1, set1 = tokenized_descs[idx_a]
        for idx_b in range(idx_a + 1, len(tokenized_descs)):
            j, c2, set2 = tokenized_descs[idx_b]
            if jaccard_similarity(set1, set2) > 0.85:
                reasons.append(f"copy_paste_fraud: high description overlap between {c1} and {c2}")
                found_fraud = True
                break

    # ── Check 5: Negative duration in a job ─────────────────────────────────
    for job in career:
        dur = job.get("duration_months", 0)
        if dur < 0:
            reasons.append(f"negative_job_duration: {job.get('company', '?')} has {dur}mo")

    # ── Check 7: Future skills in past jobs ──────────────────────────────────
    for job in career:
        desc = job.get("description", "").lower()
        end_str = job.get("end_date")
        end_year = _get_end_year(end_str) if end_str else _REFERENCE_YEAR
        for skill, year in _SKILL_RELEASE_YEARS.items():
            if skill in desc and end_year < year:
                reasons.append(f"impossible_skill_timeline: used {skill} before {year} in job ending {end_year}")
                break

    # ── Check 8: Career start < Latest Education End ─────────────────────────
    latest_edu_end = None
    for edu in education:
        end_str = edu.get("end_date")
        if end_str:
            end_date = _parse_date_cached(end_str)
            if end_date and (not latest_edu_end or end_date > latest_edu_end):
                latest_edu_end = end_date

    if career_start_dates and latest_edu_end:
        earliest_start = min(career_start_dates)
        if earliest_start < latest_edu_end:
            reasons.append(
                f"career_before_education: career started {earliest_start.year} "
                f"before education ended {latest_edu_end.year}"
            )

    return len(reasons) > 0, reasons
