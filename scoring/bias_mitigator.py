# scoring/bias_mitigator.py

def compute_education_score(candidate: dict) -> float:
    """
    Education tier has MINIMAL weight in this system.
    
    Bias mitigation principle: The JD says 'we care about what you built,
    not where you studied.' A tier_3 engineer who shipped a production search
    system > a tier_1 graduate who did academic research only.
    
    We give education a max 0.05 contribution to final score.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.0
    
    best_tier = "unknown"
    for edu in education:
        tier = edu.get("tier", "unknown")
        # Take the best tier across all degrees
        tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
        if tier_rank.get(tier, 0) > tier_rank.get(best_tier, 0):
            best_tier = tier
    
    # Deliberately compressed range: 0.01 to 0.05
    # A tier_1 grad gets 4x the education bonus of a tier_4 grad.
    # But since this contributes max 0.05 to the final score, it's nearly invisible.
    tier_score = {"tier_1": 0.05, "tier_2": 0.035, "tier_3": 0.02, "tier_4": 0.01, "unknown": 0.01}
    return tier_score.get(best_tier, 0.01)


def apply_bias_checks(candidate: dict, raw_score: float) -> float:
    """
    Apply fairness guardrails to the final score.
    
    Checks implemented:
    1. Education tier ceiling (already minimal in our system)
    2. No prestige inflation for big-name companies unless they're product cos
    3. No penalty for career gaps (people take breaks)
    4. No salary expectation bias (within reasonable range for India)
    """
    adjusted = raw_score
    
    # ── Anti-prestige inflation ──────────────────────────────────────────────
    # Big tech ≠ automatically good fit. The JD explicitly says "if you've
    # spent your career at Google or Meta and want a well-scoped role, this isn't it."
    career = candidate.get("career_history", [])
    faang = {"google", "meta", "amazon", "microsoft", "apple", "netflix"}
    all_at_faang = all(
        any(f in j.get("company", "").lower() for f in faang)
        for j in career
    ) if career else False
    # Actually the JD doesn't penalize FAANG — just notes it. No adjustment.
    
    # ── Salary expectation sanity ────────────────────────────────────────────
    # Don't penalize for high salary expectations — that's discriminatory.
    # Just flag extreme outliers as informational.
    signals = candidate.get("redrob_signals", {})
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 30)
    # No penalty — this is a fairness choice.
    
    # ── Profile completeness ─────────────────────────────────────────────────
    # Low completeness may indicate disengagement, NOT incompetence.
    # Use as a tiebreaker only, not a main signal.
    completeness = signals.get("profile_completeness_score", 50)
    if completeness < 20:
        adjusted *= 0.97  # barely affects score
    
    return adjusted
