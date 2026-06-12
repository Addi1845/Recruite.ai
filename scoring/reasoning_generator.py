# scoring/reasoning_generator.py
from datetime import datetime
from scoring.jd_config import CORE_SKILLS, PREFERRED_CITIES, EXACT_MATCH_SKILLS
import re


def _word_boundary_match(needle: str, haystack: str) -> bool:
    """Check if needle appears as a whole word in haystack."""
    pattern = r'(?<![a-zA-Z0-9])' + re.escape(needle) + r'(?![a-zA-Z0-9])'
    return bool(re.search(pattern, haystack))


def generate_reasoning(candidate: dict, breakdown: dict, rank: int) -> str:
    """
    Generate a 1-2 sentence reasoning that:
    1. References specific facts (title, YoE, actual skills used)
    2. Connects to JD requirements
    3. Mentions concerns honestly (especially for lower ranks)
    4. Does NOT hallucinate (only uses what's in the profile)
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # --- Build fact anchors from actual profile ---
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "current company")
    location = profile.get("location", "India")
    
    # Find most relevant skills (top skills by JD weight)
    ranked_skills = []
    for skill in skills:
        skill_lower = skill["name"].lower()
        for jd_skill, weight in CORE_SKILLS.items():
            if weight < 3.5:
                continue
            # Use same matching logic as skills_scorer
            if jd_skill in EXACT_MATCH_SKILLS:
                matched = _word_boundary_match(jd_skill, skill_lower)
            else:
                matched = jd_skill in skill_lower or skill_lower in jd_skill
            if matched:
                prof = skill.get("proficiency", "intermediate")
                ranked_skills.append((skill["name"], weight, prof))
                break
    ranked_skills.sort(key=lambda x: -x[1])
    top_skills = [s[0] for s in ranked_skills[:3]]
    
    # Recent activity
    last_active = signals.get("last_active_date", "")
    response_rate = signals.get("recruiter_response_rate", 0.5)
    notice = signals.get("notice_period_days", 60)
    open_to_work = signals.get("open_to_work_flag", False)
    
    # --- Build reasoning based on rank tier ---
    parts = []
    
    # Part 1: Identity + strongest fit signal
    if top_skills:
        skill_str = ", ".join(top_skills[:2])
        parts.append(
            f"{title} with {yoe:.1f}yrs exp; hands-on with {skill_str}"
        )
    else:
        parts.append(f"{title} with {yoe:.1f}yrs exp at {company}")
    
    # Part 2: Specific strengths (for top candidates)
    if rank <= 20:
        # Find production evidence
        prod_evidence_signals = ["deployed", "serving", "inference", "retrieval",
                                  "search", "ranking", "vector", "latency"]
        for job in career[:2]:
            desc = job.get("description", "").lower()
            for sig in prod_evidence_signals:
                if sig in desc:
                    parts.append(f"{sig} experience verified in career history")
                    break
            else:
                continue
            break
        
        # Location
        if any(city in location.lower() for city in PREFERRED_CITIES):
            parts.append(f"based in {location} (preferred for Pune/Noida hybrid role)")
    
    # Part 3: Concerns / honest caveats (for mid-lower ranks)
    concerns = []
    
    if notice > 90:
        concerns.append(f"{notice}d notice period (JD prefers sub-30)")
    
    if last_active:
        try:
            days = (datetime.now() - datetime.strptime(last_active, "%Y-%m-%d")).days
            if days > 90:
                concerns.append(f"inactive for {days}d (engagement risk)")
        except ValueError:
            pass
    
    if response_rate < 0.3:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    
    if not open_to_work and rank <= 30:
        concerns.append("not marked open-to-work")
    
    if breakdown.get("career", 0) < 0.3 and rank > 20:
        concerns.append("limited production ML evidence in career history")
    
    # Assemble
    main_text = "; ".join(parts[:2])
    if concerns and rank > 15:
        concern_str = ", ".join(concerns[:2])
        return f"{main_text}. Concerns: {concern_str}."
    elif concerns and rank > 50:
        return f"{main_text}. Notable concerns: {'; '.join(concerns[:2])} -- ranked here over stronger candidates."
    else:
        return f"{main_text}."
