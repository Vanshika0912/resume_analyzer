import re
import json
from typing import List, Set, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utility helpers
# -----------------------------
def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # remove common punctuation but keep words and + . - (for versions like c++ / node.js)
    text = re.sub(r"[^a-z0-9+\-_. ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_skill_dataset(path: str = "data/skills.json") -> Dict[str, List[str]]:
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------
# Skill extraction (dictionary matching)
# -----------------------------
def extract_skills_from_text(text: str, skill_list: List[str]) -> Set[str]:
    """
    Return set of skills found in text using substring match of normalized tokens.
    """
    text_norm = _normalize_text(text)
    found = set()
    for skill in skill_list:
        # normalize skill and check word boundary-ish presence
        skill_norm = _normalize_text(skill)
        if not skill_norm:
            continue
        # Check exact token match or substring (e.g., "sql" in "mysql" -> treat mysql separately)
        if f" {skill_norm} " in f" {text_norm} " or skill_norm in text_norm.split():
            found.add(skill)
        else:
            # allow substring match for multi-word skills like "machine learning"
            if skill_norm in text_norm:
                found.add(skill)
    return found

# -----------------------------
# JD skill extraction (use same function)
# -----------------------------
def extract_jd_skills(jd_text: str, skill_dict: Dict[str, List[str]]) -> Set[str]:
    tech = extract_skills_from_text(jd_text, skill_dict.get("technical_skills", []))
    soft = extract_skills_from_text(jd_text, skill_dict.get("soft_skills", []))
    return tech.union(soft)

def extract_resume_skills(resume_text: str, skill_dict: Dict[str, List[str]]) -> Set[str]:
    tech = extract_skills_from_text(resume_text, skill_dict.get("technical_skills", []))
    soft = extract_skills_from_text(resume_text, skill_dict.get("soft_skills", []))
    return tech.union(soft)

# -----------------------------
# Keyword extraction (simple, JD-driven)
# -----------------------------
def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    """
    Return top_n keywords by simple frequency after normalization.
    (Useful for keyword match %)
    """
    norm = _normalize_text(text)
    tokens = [t for t in norm.split() if len(t) > 2]  # drop tiny tokens
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_tokens[:top_n]]

# -----------------------------
# Metric calculators
# -----------------------------
def skill_match_percentage(resume_skills: Set[str], jd_skills: Set[str]) -> float:
    """
    percentage of JD skills covered by resume
    If JD has no skills extracted, returns 0.0
    """
    if not jd_skills:
        return 0.0
    matched = len(resume_skills & jd_skills)
    score = (matched / len(jd_skills)) * 100
    return round(score, 2)

def keyword_match_percentage(resume_text: str, jd_text: str, top_n: int = 25) -> float:
    jd_keywords = extract_keywords(jd_text, top_n)
    if not jd_keywords:
        return 0.0
    res_norm = _normalize_text(resume_text)
    matched = sum(1 for kw in jd_keywords if kw in res_norm)
    score = (matched / len(jd_keywords)) * 100
    return round(score, 2)

def jd_similarity_percentage(resume_text: str, jd_text: str) -> float:
    """
    TF-IDF cosine similarity between resume and JD (semantic-ish)
    Returns percentage 0..100
    """
    resume_text = resume_text or ""
    jd_text = jd_text or ""
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vecs = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return round(sim * 100, 2)
    except Exception:
        return 0.0

def final_score(skill_pct: float, keyword_pct: float, sim_pct: float,
                weights: Tuple[float, float, float] = (0.45, 0.25, 0.30)) -> float:
    """
    Weighted final score:
    default weights: skills 45%, keywords 25%, semantic similarity 30%
    """
    w_skill, w_keyword, w_sim = weights
    final = (skill_pct * w_skill) + (keyword_pct * w_keyword) + (sim_pct * w_sim)
    final = round(final, 2)
    if final < 0:
        final = 0.0
    if final > 100:
        final = 100.0
    return final

# -----------------------------
# Improved, detailed recommendations (rule-based)
# -----------------------------
def generate_detailed_recommendations(resume_text: str,
                                      resume_skills: Set[str],
                                      jd_skills: Set[str],
                                      num_example_projects: int = 3) -> Dict[str, object]:
    """
    Returns a dictionary with:
      - missing_skills: list
      - prioritized_actions: list of actions (learn/project/formatting)
      - project_templates: small templates to add to resume
      - resume_weaknesses: list of detected weaknesses with reasons
    """
    rec = {"missing_skills": [], "prioritized_actions": [], "project_templates": [], "resume_weaknesses": []}

    missing = sorted(list(jd_skills - resume_skills))
    rec["missing_skills"] = missing

    # Prioritize missing skills - first technical, then others
    if missing:
        rec["prioritized_actions"].append(
            f"Top missing skills from the JD: {', '.join(missing[:7])}. Start by learning the top 2 and building 1 small project using them."
        )
    else:
        rec["prioritized_actions"].append("You already cover the JD skills. Focus on measurable achievements and deployment details.")

    # Detect weak project descriptions (naive heuristics)
    if len(resume_text.splitlines()) < 5 or len(resume_text.split()) < 100:
        rec["resume_weaknesses"].append("Resume is very short — add at least 2-3 project bullets with technologies and measurable results.")
    # check presence of numbers/metrics
    if not re.search(r"\b\d+%|\b\d+\+?%?|\breduced\b|\bincreased\b|\bimproved\b", resume_text.lower()):
        rec["resume_weaknesses"].append("No measurable results detected — add metrics (e.g., 'reduced latency by 30%', 'improved accuracy to 92%').")
    # check for GitHub/links
    if "github" not in resume_text.lower() and "linkedin" not in resume_text.lower():
        rec["resume_weaknesses"].append("No GitHub/LinkedIn links detected — add links to code and projects.")

    # Project templates (short, ready-to-copy bullets)
    examples = []
    for i in range(num_example_projects):
        examples.append({
            "title": f"Project {i+1}: Build & Deploy {missing[0] if missing else 'Sample'} Service",
            "bullet_points": [
                "Built a REST API using Flask to expose model predictions and handle authentication.",
                "Implemented data preprocessing and feature engineering; reduced false positives by X%.",
                "Containerized the application using Docker and deployed on AWS/GCP; used CI/CD for automated releases.",
                "Hosted project repository: github.com/yourname/project (include README + demo)."
            ]
        })
    rec["project_templates"] = examples

    # Suggest resume formatting improvements
    rec["prioritized_actions"].append("Format: Use 6-8 bullet points max per role, start bullets with action verbs, include Tech Stack per project.")

    # Suggest learning resources (brief)
    if missing:
        rec["prioritized_actions"].append("Suggested learning path: (1) Official docs / quick course, (2) small project (weekend), (3) document on GitHub.")

    return rec

# -----------------------------
# Wrapper to compute all metrics given resume & jd texts and skill dict
# -----------------------------
def analyze_resume_against_jd(resume_text: str, jd_text: str, skill_dict_path: str = "data/skills.json"):
    skills = load_skill_dataset(skill_dict_path)
    jd_sk = extract_jd_skills(jd_text, skills)
    res_sk = extract_resume_skills(resume_text, skills)

    skill_pct = skill_match_percentage(res_sk, jd_sk)      # JD-driven
    keyword_pct = keyword_match_percentage(resume_text, jd_text)
    sim_pct = jd_similarity_percentage(resume_text, jd_text)
    final = final_score(skill_pct, keyword_pct, sim_pct)

    rec = generate_detailed_recommendations(resume_text, res_sk, jd_sk)

    return {
        "resume_skills": sorted(list(res_sk)),
        "jd_skills": sorted(list(jd_sk)),
        "missing_skills": rec["missing_skills"],
        "skill_pct": skill_pct,
        "keyword_pct": keyword_pct,
        "similarity_pct": sim_pct,
        "final_score": final,
        "recommendations": rec
    }
