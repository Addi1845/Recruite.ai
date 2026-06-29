# scoring/skills_scorer.py
import re
from datetime import datetime
from functools import lru_cache
from scoring.jd_config import (
    CORE_SKILLS, MAX_SKILLS_SCORE, BAD_TITLE_PATTERNS,
    BAD_TITLE_EXACT_WORDS, EXACT_MATCH_SKILLS, SKILL_SYNONYMS
)


@lru_cache(maxsize=512)
def _compile_boundary_pattern(needle: str):
    """Pre-compile and cache word-boundary regex for each unique needle.
    
    Without caching, this regex is compiled fresh for EVERY skill × EVERY candidate.
    With caching: compiled once per unique needle string (~30 patterns total).
    """
    return re.compile(r'(?<![a-zA-Z0-9])' + re.escape(needle) + r'(?![a-zA-Z0-9])')


def _word_boundary_match(needle: str, haystack: str) -> bool:
    """Check if needle appears as a whole word/phrase in haystack."""
    return bool(_compile_boundary_pattern(needle).search(haystack))


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
    
    current_year = datetime.now().year
    
    # Build a lowercase lookup of candidate skills with metadata
    skill_map = {}
    for s in skills:
        name_lower = s["name"].lower()
        
        # Skill recency multiplier
        recency_mult = 1.0
        last_used = s.get("last_used_date")
        if last_used:
            try:
                # Assuming format like "2023-05-01" or "2023"
                if isinstance(last_used, str) and len(last_used) >= 4:
                    year = int(last_used[:4])
                    age = current_year - year
                    if age >= 4:
                        recency_mult = 0.6
                    elif age >= 2:
                        recency_mult = 0.8
            except ValueError:
                pass
        
        skill_map[name_lower] = {
            "proficiency": s.get("proficiency", "beginner"),
            "endorsements": s.get("endorsements", 0),
            "duration_months": s.get("duration_months", 0),
            "recency_mult": recency_mult
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
        if jd_skill in matched_jd_skills:
            continue
            
        synonyms = [jd_skill]
        if jd_skill in SKILL_SYNONYMS:
            synonyms.extend(SKILL_SYNONYMS[jd_skill])
            
        matched_skill = None
        matched_synonym = jd_skill
        
        for syn in synonyms:
            for skill_name in skill_map:
                if syn in EXACT_MATCH_SKILLS or (jd_skill in EXACT_MATCH_SKILLS and syn == jd_skill):
                    # Use word-boundary matching for short/ambiguous JD skills
                    if _word_boundary_match(syn, skill_name):
                        matched_skill = skill_map[skill_name]
                        matched_synonym = syn
                        break
                else:
                    # Safe substring matching for longer, unambiguous terms
                    if syn in skill_name or skill_name in syn:
                        matched_skill = skill_map[skill_name]
                        matched_synonym = syn
                        break
            if matched_skill:
                for s in synonyms:
                    matched_jd_skills.add(s)
                break
        
        if matched_skill:
            prof_mult = PROFICIENCY_MULT.get(matched_skill["proficiency"], 0.5)
            
            # Duration scaling: max out at 18 months (not unfair to career changers)
            duration = matched_skill["duration_months"]
            dur_mult = min(1.0, 0.4 + 0.6 * (duration / 18)) if duration > 0 else 0.3
            
            # Recency scaling
            recency_mult = matched_skill.get("recency_mult", 1.0)
            
            # Career evidence check (critical for high-weight skills)
            evidence_mult = 1.0
            if weight >= 4.0:  # only scrutinize must-have skills
                evidence_mult = 1.0 if has_career_evidence(matched_synonym) else 0.5
            
            raw_score += weight * prof_mult * dur_mult * evidence_mult * recency_mult
        
        else:
            # Check career text for any of the synonyms
            for syn in synonyms:
                if syn in EXACT_MATCH_SKILLS or (jd_skill in EXACT_MATCH_SKILLS and syn == jd_skill):
                    if _word_boundary_match(syn, career_text):
                        raw_score += weight * 0.35
                        for s in synonyms:
                            matched_jd_skills.add(s)
                        break
                elif syn in career_text:
                    raw_score += weight * 0.35
                    for s in synonyms:
                        matched_jd_skills.add(s)
                    break
    
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
