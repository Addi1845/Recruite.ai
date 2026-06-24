# scoring/behavioral_scorer.py
from datetime import datetime
from scoring.jd_config import PREFERRED_CITIES

def compute_behavioral_score(candidate: dict) -> float:
    """
    Converts Redrob platform signals into an availability/engagement score.
    
    This is a MULTIPLIER on top of skill fit — not an additive bonus.
    A perfect-skills candidate who hasn't logged in for 6 months and has
    5% response rate is, for hiring purposes, NOT available.
    """
    signals = candidate.get("redrob_signals", {})
    score = 0.5  # neutral baseline
    
    # ── Recency: When did they last engage? (MOST IMPORTANT signal) ──────────
    try:
        last_active = datetime.strptime(signals["last_active_date"], "%Y-%m-%d")
        REFERENCE_DATE = datetime(2025, 6, 1)
        days_inactive = (REFERENCE_DATE - last_active).days
        
        if days_inactive <= 7:
            score += 0.30
        elif days_inactive <= 30:
            score += 0.20
        elif days_inactive <= 60:
            score += 0.10
        elif days_inactive <= 90:
            score += 0.0
        elif days_inactive <= 180:
            score -= 0.15
        else:
            score -= 0.30  # >6 months inactive: major red flag
    except (KeyError, ValueError):
        score -= 0.05
    
    # ── Open to work flag ─────────────────────────────────────────────────────
    if signals.get("open_to_work_flag", False):
        score += 0.12
    
    # ── Recruiter response rate ───────────────────────────────────────────────
    response_rate = signals.get("recruiter_response_rate", 0.5)
    # Normalize around 0.5: below = penalty, above = bonus
    score += (response_rate - 0.5) * 0.25
    
    # ── Response speed ────────────────────────────────────────────────────────
    avg_response_hours = signals.get("avg_response_time_hours", 24)
    if avg_response_hours <= 4:
        score += 0.06
    elif avg_response_hours <= 24:
        score += 0.02
    elif avg_response_hours > 72:
        score -= 0.06
    
    # ── Notice period ─────────────────────────────────────────────────────────
    # JD says: "would love sub-30-day notice; can buy out 30 days"
    notice = signals.get("notice_period_days", 60)
    if notice <= 15:
        score += 0.12
    elif notice <= 30:
        score += 0.08
    elif notice <= 60:
        score += 0.03
    elif notice <= 90:
        score -= 0.03
    else:
        score -= 0.10
    
    # ── GitHub activity ───────────────────────────────────────────────────────
    github = signals.get("github_activity_score", -1)
    if github == -1:
        pass  # neutral — not penalizing for no GitHub
    elif github >= 70:
        score += 0.10
    elif github >= 40:
        score += 0.05
    elif github >= 10:
        score += 0.01
    
    # ── Market validation: recruiters are already interested ──────────────────
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        score += 0.05
    elif saved >= 2:
        score += 0.02
    
    # ── Reliability: shows up when scheduled ─────────────────────────────────
    interview_rate = signals.get("interview_completion_rate", 0.7)
    if interview_rate >= 0.9:
        score += 0.05
    elif interview_rate < 0.5:
        score -= 0.10
    
    # ── Location fit ─────────────────────────────────────────────────────────
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    
    if any(city in location for city in PREFERRED_CITIES):
        score += 0.10
    elif country == "india" and willing_to_relocate:
        score += 0.05
    elif country != "india" and not willing_to_relocate:
        score -= 0.15  # outside India, not willing to relocate
    
    return min(1.0, max(0.0, score))
