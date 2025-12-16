import spacy
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# -----------------------------------------------------
# Load skill dataset
# -----------------------------------------------------
def load_skill_dataset():
    with open("data/skills.json", "r") as f:
        return json.load(f)

# -----------------------------------------------------
# Extract Email
# -----------------------------------------------------
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    return match.group(0) if match else None

# -----------------------------------------------------
# Extract Skills
# -----------------------------------------------------
def extract_skills(text, skill_list):
    found = []
    for skill in skill_list:
        if skill.lower() in text.lower():
            found.append(skill)
    return list(set(found))

# -----------------------------------------------------
# Keyword Match
# -----------------------------------------------------
def keyword_match(resume_text, job_description):
    resume_words = set(resume_text.lower().split())
    jd_words = set(job_description.lower().split())

    if len(jd_words) == 0:
        return 0

    match = len(resume_words & jd_words) / len(jd_words)
    return round(match * 100, 2)

# -----------------------------------------------------
# Semantic Similarity (TF-IDF)
# -----------------------------------------------------
def similarity_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(vectors[0], vectors[1])
        return round(score[0][0] * 100, 2)
    except:
        return 0.0

# -----------------------------------------------------
# Recommendations generator
# -----------------------------------------------------
def generate_recommendations(found_skills, missing_skills):
    suggestions = []

    if missing_skills:
        suggestions.append(
            f"Consider learning or adding these missing skills: {', '.join(missing_skills)}."
        )

    if len(found_skills) < 5:
        suggestions.append("Try adding more measurable achievements in your resume.")

    if not suggestions:
        suggestions.append("Your resume looks strong! Just tailor it to specific job descriptions.")

    return "\n".join(suggestions)