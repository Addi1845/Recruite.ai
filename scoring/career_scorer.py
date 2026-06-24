# scoring/career_scorer.py
from scoring.jd_config import (
    CONSULTING_FIRMS, PRODUCTION_SIGNALS, RESEARCH_ONLY_SIGNALS,
    GOOD_TITLES, BAD_TITLE_PATTERNS, ML_CONTEXT_SIGNALS
)

def _has_ml_context(desc: str) -> bool:
    """Check if a job description has ML/AI context.
    
    This prevents manufacturing 'production' or hardware 'production tooling'
    from being counted as ML production deployment evidence.
    """
    return any(ctx in desc for ctx in ML_CONTEXT_SIGNALS)


def _count_production_signals(desc: str) -> int:
    """Count production signals, but only in ML-relevant contexts.
    
    The word 'production' by itself is too ambiguous (manufacturing, media
    production, etc.). We only count it when ML context words are present.
    """
    hits = sum(1 for sig in PRODUCTION_SIGNALS if sig in desc)
    
    # "production" alone is ambiguous — only count it if ML context is present
    if "production" in desc and _has_ml_context(desc):
        hits += 1
    
    return hits


def compute_career_score(candidate: dict) -> float:
    """
    Evaluates career history quality for this specific JD.
    
    Key insight: a 6-year engineer who built a search ranking system at a
    Series B startup scores higher than a 9-year engineer who did Java CRUD
    at Infosys and recently added AI keywords to their profile.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0
    
    score = 0.0
    
    # ── Company quality check ────────────────────────────────────────────────
    all_companies = [j.get("company", "").lower() for j in career]
    consulting_jobs = [c for c in all_companies if any(firm in c for firm in CONSULTING_FIRMS)]
    product_company_jobs = len(career) - len(consulting_jobs)
    
    # Consulting jobs check: graduated penalty based on percentage
    total_jobs = len(career)
    if total_jobs > 0:
        consulting_ratio = len(consulting_jobs) / total_jobs
        if consulting_ratio > 0:
            score -= 0.35 * consulting_ratio
    
    if product_company_jobs >= 2:
        score += 0.10
    
    # ── Per-role analysis ────────────────────────────────────────────────────
    has_production_ml = False
    has_research_only = False
    has_relevant_title = False
    
    for job in career:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        company = job.get("company", "").lower()
        duration = job.get("duration_months", 0)
        
        # Skip very short roles (<2 months) — not meaningful signal
        if duration < 2:
            continue
        
        # Is this a relevant AI/ML title?
        if any(t in title for t in GOOD_TITLES):
            has_relevant_title = True
            # Bonus for longer tenure in relevant roles (shows depth, not title-chasing)
            tenure_bonus = min(0.08, duration / 600)  # max at 50 months
            score += tenure_bonus
        
        # Production deployment evidence in description (ML-context-aware)
        prod_hits = _count_production_signals(desc)
        if prod_hits >= 3:
            has_production_ml = True
            score += 0.08
        elif prod_hits >= 1:
            score += 0.03
        
        # Research-only signals (penalize only if NO production evidence exists)
        research_hits = sum(1 for sig in RESEARCH_ONLY_SIGNALS if sig in desc)
        if research_hits >= 2 and prod_hits == 0:
            has_research_only = True
        
        # Role-specific high-value signals (building search/ranking/retrieval)
        hv_signals = [
            "ranking", "retrieval", "recommendation", "search", "embedding",
            "vector", "matching", "similarity", "nlp", "text mining",
            "information retrieval", "semantic"
        ]
        hv_hits = sum(1 for sig in hv_signals if sig in desc)
        score += min(0.12, hv_hits * 0.015)
    
    if has_production_ml:
        score += 0.15
    
    if has_research_only and not has_production_ml:
        score -= 0.20
    
    if has_relevant_title:
        score += 0.10
        
    # ── Leadership check ─────────────────────────────────────────────────────
    leadership_keywords = ["led team", "managed", "mentored", "tech lead", "architect", "staff engineer"]
    leadership_hits = sum(1 for job in career if any(kw in str(job.get("description", "")).lower() for kw in leadership_keywords))
    score += min(0.06, leadership_hits * 0.03)
    
    # ── Title trajectory check ───────────────────────────────────────────────
    # Red flag: title_chaser (short stints at multiple companies for title bumps)
    if len(career) >= 4:
        avg_duration = sum(j.get("duration_months", 0) for j in career) / len(career)
        if avg_duration < 14:  # <14 months average = title hopper
            score -= 0.10
    
    return min(1.0, max(0.0, score))
