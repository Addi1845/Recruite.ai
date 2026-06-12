# scoring/skills_scorer.py
import re
from scoring.jd_config import (
    CORE_SKILLS, MAX_SKILLS_SCORE, BAD_TITLE_PATTERNS,
    BAD_TITLE_EXACT_WORDS, EXACT_MATCH_SKILLS
)


def _word_boundary_match(needle: str, haystack: str) -> bool:
    """Check if needle appears as a whole word/phrase in haystack.
    
    Prevents 'ml' from matching 'html' or 'xml',
    and 'map' from matching 'roadmap' or 'bitmap'.
    """
    pattern = r'(?<![a-zA-Z0-9])' + re.escape(needle) + r'(?![a-zA-Z0-9])'
    return bool(re.search(pattern, haystack))


def _is_bad_title(title_lower: str) -> bool:
    """Check if title matches bad patterns using appropriate matching strategy."""
    # Substring patterns (longer, safe from false positives)
    if any(bad in title_lower for bad in BAD_TITLE_PATTERNS):
        return True
    # Short patterns need word-boundary matching
    # "hr" should match "hr manager" but not "scheduler" or "chrome"
    for word in BAD_TITLE_EXACT_WORDS:
        if _word_boundary_match(word, title_lower):
            return True
    return False


def compute_skills_score(candidate: dict) -> float:
    """Returns 0-1 score for skill-JD fit"""
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    
    if not skills:
        return 0.0
    
    # Build a lowercase lookup of candidate skills with metadata
    skill_map = {}
    for s in skills:
        name_lower = s["name"].lower()
        skill_map[name_lower] = {
            "proficiency": s.get("proficiency", "beginner"),
            "endorsements": s.get("endorsements", 0),
            "duration_months": s.get("duration_months", 0)
        }
    
    # All career text (titles + descriptions)
    career_text = " ".join(
        (j.get("title", "") + " " + j.get("description", "")).lower()
        for j in career
    ).strip()
    
    # Also include summary and headline for evidence checking
    profile = candidate.get("profile", {})
    full_text = career_text + " " + profile.get("summary", "").lower() + " " + profile.get("headline", "").lower()
    
    # ── Coherence check: are high-value AI skills backed by career evidence? ──
    def has_career_evidence(skill_name: str) -> bool:
        """Check if the skill has backing in career history or profile.
        
        For short skill names (<=3 chars), use word-boundary matching
        to prevent 'ml' from matching 'html'.
        """
        keywords = skill_name.split()
        for kw in keywords:
            if len(kw) <= 3:
                # Use word-boundary match for short keywords
                if _word_boundary_match(kw, full_text):
                    return True
            elif len(kw) > 3 and kw in full_text:
                return True
        return False
    
    # ── Title coherence check ────────────────────────────────────────────────
    current_title = candidate["profile"]["current_title"].lower()
    is_mismatched_title = _is_bad_title(current_title)
    title_penalty = 0.6 if is_mismatched_title else 1.0
    
    # ── Core score computation ───────────────────────────────────────────────
    raw_score = 0.0
    matched_jd_skills = set()  # Track to avoid double-counting synonyms
    
    PROFICIENCY_MULT = {
        "beginner": 0.4,
        "intermediate": 0.7,
        "advanced": 1.0,
        "expert": 1.15
    }
    
    for jd_skill, weight in CORE_SKILLS.items():
        # Skip if we already matched a synonym of this skill
        # (e.g., "embedding" and "embeddings" shouldn't double-count)
        
        # Direct match in skills list
        matched_skill = None
        for skill_name in skill_map:
            if jd_skill in EXACT_MATCH_SKILLS:
                # Use word-boundary matching for short/ambiguous JD skills
                if _word_boundary_match(jd_skill, skill_name):
                    matched_skill = skill_map[skill_name]
                    break
            else:
                # Safe substring matching for longer, unambiguous terms
                if jd_skill in skill_name or skill_name in jd_skill:
                    matched_skill = skill_map[skill_name]
                    break
        
        if matched_skill:
            prof_mult = PROFICIENCY_MULT.get(matched_skill["proficiency"], 0.5)
            
            # Duration scaling: max out at 18 months (not unfair to career changers)
            duration = matched_skill["duration_months"]
            dur_mult = min(1.0, 0.4 + 0.6 * (duration / 18)) if duration > 0 else 0.3
            
            # Career evidence check (critical for high-weight skills)
            evidence_mult = 1.0
            if weight >= 4.0:  # only scrutinize must-have skills
                evidence_mult = 1.0 if has_career_evidence(jd_skill) else 0.5
            
            raw_score += weight * prof_mult * dur_mult * evidence_mult
        
        elif jd_skill in EXACT_MATCH_SKILLS:
            # For short skills, use word-boundary check in career text too
            if _word_boundary_match(jd_skill, career_text):
                raw_score += weight * 0.35
        elif jd_skill in career_text:
            # Mentioned in career but not listed in skills → partial credit
            raw_score += weight * 0.35
    
    # Apply title coherence penalty
    final = (raw_score / max(MAX_SKILLS_SCORE, 1.0)) * title_penalty
    
    return min(1.0, max(0.0, final))


def compute_assessment_bonus(candidate: dict) -> float:
    """Objective skill assessments on the Redrob platform → small bonus"""
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if not assessments:
        return 0.0
    
    relevant_keys = ["python", "ml", "nlp", "retrieval", "embedding", "search",
                     "ranking", "vector", "transformers", "pytorch", "tensorflow"]
    relevant_scores = []
    
    for skill_name, score in assessments.items():
        skill_lower = skill_name.lower()
        if any(k in skill_lower for k in relevant_keys):
            relevant_scores.append(score)
    
    if not relevant_scores:
        return 0.0
    
    avg = sum(relevant_scores) / len(relevant_scores)
    return (avg / 100) * 0.08  # max 0.08 bonus
