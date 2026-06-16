# scoring/advanced_features.py
"""
Advanced scoring features for candidate ranking.

All functions accept a candidate dict and return a float in [0.0, 1.0].
Pure Python — no sklearn, no network calls, no external APIs.
"""

from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AI_DOMAIN_KEYWORDS = frozenset({
    "ml", "machine learning", "ai", "artificial intelligence",
    "nlp", "natural language", "search", "ranking", "retrieval",
    "embedding", "vector", "transformer", "neural", "deep learning",
    "recommendation", "classification", "regression", "clustering",
    "pytorch", "tensorflow", "bert", "gpt", "llm",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "xgboost", "lightgbm", "scikit", "scipy", "numpy", "pandas", "python",
    "data science", "feature engineering", "model", "inference", "fine-tuning",
    "rag", "bm25", "ndcg", "mrr", "information retrieval",
    "semantic search", "sentence transformers", "dense retrieval",
    "hybrid search", "learning to rank", "ltr", "reranking",
})

# Maps title keywords → numeric seniority level.
# Checked via substring match against lowercased job title.
_SENIORITY_MAP = {
    "intern": 0,
    "junior": 1,
    "associate": 1,
    "mid": 2,
    "senior": 3,
    "staff": 4,
    "lead": 4,
    "principal": 5,
    "director": 5,
    "vp": 6,
    "vice president": 6,
}

# Keywords that indicate an ML/AI role (used by trajectory scorer).
_ML_ROLE_KEYWORDS = frozenset({
    "ml", "machine learning", "ai", "artificial intelligence",
    "data scientist", "data science", "nlp", "deep learning",
    "search engineer", "ranking", "retrieval", "computer vision",
    "research scientist", "applied scientist",
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Skill Credibility
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_credibility(candidate: dict) -> float:
    """Score how credible a candidate's listed skills are (0.0–1.0).

    Credibility heuristic per skill:
    - High endorsements (>5) AND reasonable duration (>6 months) → very credible.
    - Expert proficiency with 0 endorsements AND short duration (<3 months) →
      suspicious; low credibility.
    - Everything else falls on a smooth gradient between those anchors.

    Returns the average credibility across all skills, or 0.5 (neutral) if
    the candidate has no skills data.
    """
    try:
        skills = candidate.get("skills", [])
        if not skills:
            return 0.5  # no signal → neutral default

        credibility_scores: list[float] = []

        for skill in skills:
            try:
                endorsements = max(0, int(skill.get("endorsements", 0)))
                duration = max(0, int(skill.get("duration_months", 0)))
                proficiency = str(skill.get("proficiency", "")).lower()

                cred = 0.5  # baseline

                # ── Positive signals ─────────────────────────────────
                if endorsements > 5 and duration > 6:
                    # Well-endorsed AND sustained → high credibility
                    cred = 0.9
                elif endorsements > 2 and duration > 3:
                    cred = 0.75
                elif endorsements > 0 and duration > 0:
                    cred = 0.6

                # ── Suspicious patterns ──────────────────────────────
                if proficiency == "expert" and endorsements == 0 and duration < 3:
                    # Claims expert but nobody endorses it and barely used
                    cred = 0.15
                elif proficiency == "expert" and endorsements == 0:
                    # Expert, no endorsements, but at least has duration
                    cred = min(cred, 0.4)
                elif proficiency == "advanced" and endorsements == 0 and duration == 0:
                    cred = 0.25

                # ── Duration bonus (diminishing returns) ─────────────
                # Extra nudge for long-tenured skills even if endorsements
                # are merely okay.
                if duration >= 24:
                    cred = min(1.0, cred + 0.1)
                elif duration >= 12:
                    cred = min(1.0, cred + 0.05)

                credibility_scores.append(cred)

            except (TypeError, ValueError):
                # Malformed single skill → treat as neutral
                credibility_scores.append(0.5)

        if not credibility_scores:
            return 0.5

        avg = sum(credibility_scores) / len(credibility_scores)
        return round(min(1.0, max(0.0, avg)), 4)

    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Specialization Score
# ─────────────────────────────────────────────────────────────────────────────

def _skill_is_ai_domain(skill_name: str) -> bool:
    """Return True if *skill_name* belongs to the AI/ML/Search domain.

    Uses a two-pass check:
    1. Exact membership in AI_DOMAIN_KEYWORDS (handles multi-word phrases).
    2. Substring scan — any keyword that appears as a component of the skill
       name counts. This catches "sentence-transformers" matching
       "sentence transformers", or "pytorch lightning" matching "pytorch".

    Short keywords (<=3 chars) use word-boundary-aware logic to avoid
    "ml" matching "html".
    """
    name_lower = skill_name.lower().strip()
    if not name_lower:
        return False

    # Pass 1: exact match
    if name_lower in AI_DOMAIN_KEYWORDS:
        return True

    # Pass 2: substring / component match
    for kw in AI_DOMAIN_KEYWORDS:
        if len(kw) <= 3:
            # Word-boundary check for short keywords.
            # Check if kw appears surrounded by non-alphanumeric (or edges).
            idx = name_lower.find(kw)
            while idx != -1:
                left_ok = (idx == 0 or not name_lower[idx - 1].isalnum())
                right_end = idx + len(kw)
                right_ok = (right_end >= len(name_lower)
                            or not name_lower[right_end].isalnum())
                if left_ok and right_ok:
                    return True
                idx = name_lower.find(kw, idx + 1)
        else:
            if kw in name_lower:
                return True

    return False


def compute_specialization_score(candidate: dict) -> float:
    """Score how specialised the candidate is toward AI/ML/Search (0.0–1.0).

    Ratio mapping:
        >=0.70 → 1.0  (specialist)
        0.50–0.70 → linearly interpolated 0.65–1.0
        0.30–0.50 → linearly interpolated 0.30–0.65
        <0.30  → 0.30 (generalist)
        No skills → 0.3 (assume generalist)
    """
    try:
        skills = candidate.get("skills", [])
        if not skills:
            return 0.3

        total = len(skills)
        ai_count = sum(1 for s in skills if _skill_is_ai_domain(s.get("name", "")))

        ratio = ai_count / total

        if ratio >= 0.70:
            return 1.0
        elif ratio >= 0.50:
            # Linear interpolation: 0.50→0.65, 0.70→1.0
            return round(0.65 + (ratio - 0.50) / 0.20 * 0.35, 4)
        elif ratio >= 0.30:
            # Linear interpolation: 0.30→0.30, 0.50→0.65
            return round(0.30 + (ratio - 0.30) / 0.20 * 0.35, 4)
        else:
            return 0.3

    except Exception:
        return 0.3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Career Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def _extract_seniority(title: str) -> int | None:
    """Return numeric seniority from a job title, or None if undetectable."""
    title_lower = title.lower()
    # Check longer phrases first to avoid "vice president" being overridden
    # by "president" (which we don't have, but defensive).
    for keyword in sorted(_SENIORITY_MAP, key=len, reverse=True):
        if keyword in title_lower:
            return _SENIORITY_MAP[keyword]
    return None


def _is_ml_role(title: str, description: str = "") -> bool:
    """Return True if the job role is ML/AI-related."""
    text = (title + " " + description).lower()
    return any(kw in text for kw in _ML_ROLE_KEYWORDS)


def _parse_date_safe(date_str: str) -> datetime | None:
    """Parse YYYY-MM-DD date, returning None on failure."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def compute_career_trajectory(candidate: dict) -> float:
    """Score upward career trajectory toward AI/ML (0.0–1.0).

    Components:
    1. **Seniority progression** – sorted by start_date, do seniority levels
       go up, stay flat, or go down?
       - Net upward steps → bonus (up to +0.3)
       - Flat → neutral (0.0)
       - Net downward steps → slight penalty (capped at -0.15)

    2. **Non-ML → ML pivot** – if early roles are non-ML and later roles are
       ML, that shows intentional career shift → bonus (+0.2).

    Base score is 0.5 (neutral); adjustments added/subtracted from there.
    """
    try:
        career = candidate.get("career_history", [])
        if not career:
            return 0.5  # no signal

        # Sort by start_date ascending (earliest first)
        dated_roles: list[tuple[datetime, dict]] = []
        for job in career:
            dt = _parse_date_safe(job.get("start_date", ""))
            if dt is not None:
                dated_roles.append((dt, job))

        if not dated_roles:
            return 0.5

        dated_roles.sort(key=lambda x: x[0])

        # ── Seniority progression ────────────────────────────────────────────
        seniority_levels: list[int] = []
        for _, job in dated_roles:
            level = _extract_seniority(job.get("title", ""))
            if level is not None:
                seniority_levels.append(level)

        trajectory_adj = 0.0
        if len(seniority_levels) >= 2:
            # Count net steps between consecutive detected levels
            net_steps = 0
            for i in range(1, len(seniority_levels)):
                diff = seniority_levels[i] - seniority_levels[i - 1]
                net_steps += diff

            if net_steps > 0:
                # Upward trajectory — cap bonus at 0.3
                trajectory_adj = min(0.3, net_steps * 0.1)
            elif net_steps < 0:
                # Downward trajectory — gentle penalty, capped at -0.15
                trajectory_adj = max(-0.15, net_steps * 0.05)
            # net_steps == 0: flat, no adjustment

        # ── Non-ML → ML career pivot ────────────────────────────────────────
        pivot_bonus = 0.0
        if len(dated_roles) >= 2:
            # Split career into first half and second half
            mid = len(dated_roles) // 2
            early_roles = dated_roles[:mid]
            later_roles = dated_roles[mid:]

            early_ml = any(
                _is_ml_role(job.get("title", ""), job.get("description", ""))
                for _, job in early_roles
            )
            later_ml = any(
                _is_ml_role(job.get("title", ""), job.get("description", ""))
                for _, job in later_roles
            )

            if not early_ml and later_ml:
                # Intentional pivot into ML/AI
                pivot_bonus = 0.2
            elif early_ml and later_ml:
                # Consistent ML career — small bonus
                pivot_bonus = 0.1

        score = 0.5 + trajectory_adj + pivot_bonus
        return round(min(1.0, max(0.0, score)), 4)

    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 4. Salary Fit
# ─────────────────────────────────────────────────────────────────────────────

# Budget range for "Senior AI Engineer – Search & Ranking" in LPA (INR)
_BUDGET_MIN_LPA = 25.0
_BUDGET_MAX_LPA = 55.0


def _parse_salary_range(salary_str: str) -> tuple[float | None, float | None]:
    """Extract (min_lpa, max_lpa) from a salary string.

    Handles formats like:
    - "30-45 LPA"
    - "30.0 - 45.0"
    - "40 LPA"  (single number → treat as both min and max)
    - "30-45"   (no unit)
    """
    if not salary_str or not isinstance(salary_str, str):
        return None, None

    # Strip common units / suffixes
    cleaned = salary_str.lower().replace("lpa", "").replace("inr", "").strip()
    cleaned = cleaned.replace(",", "")

    parts = [p.strip() for p in cleaned.split("-") if p.strip()]

    try:
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        elif len(parts) == 1:
            val = float(parts[0])
            return val, val
    except ValueError:
        pass

    return None, None


def compute_salary_fit(candidate: dict) -> float:
    """Score salary-budget alignment (0.0–1.0).

    Budget for this role: 25–55 LPA (INR).

    Scoring bands (based on candidate's expected range midpoint):
        25–55 LPA  → 1.0  (perfect fit)
        15–25 LPA  → 0.8  (possibly junior but willing)
        55–80 LPA  → 0.6  (possibly overqualified / expensive)
        >80 or <15 → 0.4  (significant mismatch)
        No data    → 0.7  (neutral / benefit of the doubt)
    """
    try:
        redrob = candidate.get("redrob_signals", {})
        if not redrob:
            return 0.7

        salary_raw = redrob.get("expected_salary_range_inr_lpa")

        # Handle the field being a string, a number, or absent
        if salary_raw is None:
            return 0.7

        # If it's already a numeric type (int/float), treat as single value
        if isinstance(salary_raw, (int, float)):
            candidate_min = float(salary_raw)
            candidate_max = float(salary_raw)
        else:
            candidate_min, candidate_max = _parse_salary_range(str(salary_raw))

        if candidate_min is None or candidate_max is None:
            return 0.7

        # Use midpoint for band classification
        midpoint = (candidate_min + candidate_max) / 2.0

        if _BUDGET_MIN_LPA <= midpoint <= _BUDGET_MAX_LPA:
            return 1.0
        elif 15.0 <= midpoint < _BUDGET_MIN_LPA:
            return 0.8
        elif _BUDGET_MAX_LPA < midpoint <= 80.0:
            return 0.6
        else:
            # midpoint > 80 or midpoint < 15
            return 0.4

    except Exception:
        return 0.7
