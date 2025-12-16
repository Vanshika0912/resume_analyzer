import streamlit as st
import os
from utils.extract_text import extract_text_from_pdf, extract_text_from_docx
from utils.nlp_processing import extract_email, extract_skills, load_skill_dataset
from utils.nlp_processing import keyword_match, similarity_score, generate_recommendations
from utils.scoring import calculate_final_score

# ---------------- GEMINI API ----------------
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyBe0Ps1rle_Cn_wNIOxeUqK1lqBVgAaG9M"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

else:
    gemini_model = None

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

st.title("📄 Smart Resume Analyzer – AI Resume Evaluation System")

st.markdown("""
Upload your resume and paste the job description to get:

✅ Resume–JD Match %  
✅ Skill Match %  
✅ AI-Based Resume Score  
✅ Missing Skill Suggestions  
✅ AI Summary & Improvements  
""")

uploaded = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description Here", height=200)

# Load skill sets
skill_data = load_skill_dataset()
tech_skills = skill_data["technical_skills"]
soft_skills = skill_data["soft_skills"]

# ----------------------------------------------------
# PROCESSING LOGIC
# ----------------------------------------------------

if uploaded and job_description.strip():

    ext = uploaded.name.split(".")[-1]

    # Extract text
    if ext == "pdf":
        resume_text = extract_text_from_pdf(uploaded)
    else:
        resume_text = extract_text_from_docx(uploaded)

    if not resume_text:
        st.error("❌ Could not extract text from the resume. Try uploading a cleaner version.")
        st.stop()

    st.subheader("🔍 Resume Preview (First 500 characters)")
    st.write(resume_text[:500] + "...")

    # Extract key info
    email = extract_email(resume_text)

    # Skill Matching
    found_tech = extract_skills(resume_text, tech_skills)
    found_soft = extract_skills(resume_text, soft_skills)

    skill_score = (len(found_tech) / len(tech_skills)) * 100

    # JD Matching Scores
    keyword_score = keyword_match(resume_text, job_description)
    sim_score = similarity_score(resume_text, job_description)

    # Final Score
    final_score = calculate_final_score(
        skill_score=skill_score,
        jd_score=sim_score,
        keyword_score=keyword_score
    )

    # Missing Skills
    missing_skills = list(set(tech_skills) - set(found_tech))

    # Recommendations
    suggestions = generate_recommendations(found_tech, missing_skills)

    # ----------- AI RESUME EVALUATION (Gemini) -----------
    ai_feedback = ""
    if gemini_model:
        with st.spinner("🤖 Gemini AI analyzing your resume..."):
            prompt = f"""
            You are an ATS Resume Analyzer.
            Analyze the resume below for the job description and give:

            1. Profile Summary  
            2. Major strengths  
            3. Missing skills  
            4. JD Alignment Analysis  
            5. Improvements needed  
            6. Rewrite an optimized resume summary (strong ATS style)

            --- RESUME ---
            {resume_text}

            --- JOB DESCRIPTION ---
            {job_description}
            """

            ai_feedback = gemini_model.generate_content(prompt).text

    # ----------------------------------------------------
    # RESULTS
    # ----------------------------------------------------

    st.subheader("📊 Analysis Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Skill Match %", f"{skill_score:.2f}%")
    col2.metric("Keyword Match %", f"{keyword_score:.2f}%")
    col3.metric("JD Similarity %", f"{sim_score:.2f}%")

    st.metric("⭐ Final Resume Score", f"{final_score}")

    st.subheader("🧠 Skills Identified")
    st.write(found_tech)

    st.subheader("❌ Missing Skills")
    st.write(missing_skills)

    st.subheader("📝 System Recommendations")
    st.write(suggestions)

    if gemini_model:
        st.subheader("🤖 AI-Powered Resume Insights (Gemini)")
        st.markdown(ai_feedback)

    st.success("✔️ Analysis completed!")

