"""Quick smoke test for scoring.advanced_features"""
import sys
sys.path.insert(0, ".")

from scoring.advanced_features import (
    compute_skill_credibility,
    compute_specialization_score,
    compute_career_trajectory,
    compute_salary_fit,
)

errors = []

def check(name, actual, expected_min, expected_max):
    ok = expected_min <= actual <= expected_max
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name} = {actual:.3f}  (expected {expected_min}-{expected_max})")
    if not ok:
        errors.append(name)

print("=== 1. Skill Credibility ===")
credible = {"skills": [
    {"name": "Python", "proficiency": "expert", "endorsements": 12, "duration_months": 36},
    {"name": "PyTorch", "proficiency": "advanced", "endorsements": 8, "duration_months": 18},
]}
suspicious = {"skills": [
    {"name": "TensorFlow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    {"name": "BERT", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
    {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 2},
]}
check("credible_candidate", compute_skill_credibility(credible), 0.8, 1.0)
check("suspicious_candidate", compute_skill_credibility(suspicious), 0.0, 0.35)
check("empty_candidate", compute_skill_credibility({}), 0.4, 0.6)

print("\n=== 2. Specialization Score ===")
specialist = {"skills": [
    {"name": "PyTorch"}, {"name": "FAISS"}, {"name": "NLP"},
    {"name": "Ranking"}, {"name": "Embeddings"}, {"name": "Python"},
    {"name": "Search"}, {"name": "Transformer"},
]}
generalist = {"skills": [
    {"name": "Java"}, {"name": "Spring Boot"}, {"name": "React"},
    {"name": "CSS"}, {"name": "HTML"}, {"name": "Docker"},
    {"name": "Kubernetes"}, {"name": "Python"},
]}
check("specialist", compute_specialization_score(specialist), 0.9, 1.0)
check("generalist", compute_specialization_score(generalist), 0.25, 0.45)
check("no_skills", compute_specialization_score({}), 0.25, 0.35)

# Edge: "HTML" should NOT match "ml" keyword
html_only = {"skills": [{"name": "HTML"}, {"name": "CSS"}, {"name": "JavaScript"}]}
check("html_not_ml", compute_specialization_score(html_only), 0.0, 0.35)

print("\n=== 3. Career Trajectory ===")
upward_pivot = {"career_history": [
    {"title": "Junior Software Eng", "description": "Java CRUD", "start_date": "2016-01-01"},
    {"title": "Software Engineer", "description": "Backend APIs", "start_date": "2018-06-01"},
    {"title": "Senior ML Engineer", "description": "Built search ranking system", "start_date": "2021-01-01"},
    {"title": "Lead ML Engineer", "description": "Leading AI retrieval team", "start_date": "2023-06-01"},
]}
flat = {"career_history": [
    {"title": "Software Engineer", "description": "Web dev", "start_date": "2018-01-01"},
    {"title": "Software Engineer", "description": "Web dev 2", "start_date": "2020-01-01"},
]}
downward = {"career_history": [
    {"title": "Lead Engineer", "description": "Led team", "start_date": "2018-01-01"},
    {"title": "Junior Developer", "description": "Coding", "start_date": "2021-01-01"},
]}
check("upward_pivot", compute_career_trajectory(upward_pivot), 0.7, 1.0)
check("flat_career", compute_career_trajectory(flat), 0.4, 0.65)
check("downward", compute_career_trajectory(downward), 0.2, 0.5)
check("no_career", compute_career_trajectory({}), 0.45, 0.55)

print("\n=== 4. Salary Fit ===")
check("perfect_fit", compute_salary_fit({"redrob_signals": {"expected_salary_range_inr_lpa": "30-45 LPA"}}), 1.0, 1.0)
check("slightly_cheap", compute_salary_fit({"redrob_signals": {"expected_salary_range_inr_lpa": "18-22 LPA"}}), 0.8, 0.8)
check("expensive", compute_salary_fit({"redrob_signals": {"expected_salary_range_inr_lpa": "60-75 LPA"}}), 0.6, 0.6)
check("way_off", compute_salary_fit({"redrob_signals": {"expected_salary_range_inr_lpa": "90-120 LPA"}}), 0.4, 0.4)
check("no_salary", compute_salary_fit({"redrob_signals": {}}), 0.7, 0.7)
check("numeric_val", compute_salary_fit({"redrob_signals": {"expected_salary_range_inr_lpa": 40}}), 1.0, 1.0)
check("malformed", compute_salary_fit({"redrob_signals": {"expected_salary_range_inr_lpa": "not-a-number"}}), 0.7, 0.7)
check("no_signals", compute_salary_fit({}), 0.7, 0.7)

print(f"\n{'='*50}")
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
