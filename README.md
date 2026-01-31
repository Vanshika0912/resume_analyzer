# Smart Resume Analyzer

An AI-powered resume analysis tool that evaluates resumes against job descriptions using natural language processing and machine learning techniques. The application provides comprehensive insights including skill matching, keyword analysis, similarity scoring, and AI-generated recommendations for resume improvement.

## Features

- **Resume Upload**: Support for PDF and DOCX resume formats
- **Job Description Analysis**: Paste job descriptions for comparison
- **Skill Extraction**: Automatic identification of technical and soft skills from resumes
- **Match Scoring**: Calculate skill match percentage, keyword match, and overall similarity
- **AI Insights**: Integration with Google Gemini AI for detailed resume evaluation and suggestions
- **Recommendations**: System-generated suggestions for missing skills and improvements
- **Web Interface**: User-friendly Streamlit-based web application

## Technologies Used

- **Frontend**: Streamlit
- **AI/ML**: Google Generative AI (Gemini), spaCy, NLTK, scikit-learn
- **Document Processing**: pdfplumber, python-docx, PyPDF2
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Database**: SQLAlchemy
- **Other**: regex, tqdm, pydantic, reportlab

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd resume_analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Gemini API:
   - Obtain an API key from Google AI Studio
   - Replace the `GEMINI_API_KEY` in `app.py` with your API key

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the provided URL (usually `http://localhost:8501`)

3. Upload a resume (PDF or DOCX) and paste the job description

4. Click analyze to get comprehensive resume evaluation results

## Project Structure

```
resume_analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   └── skills.json        # Skills dataset
├── db/                    # Database files
├── models/                # ML models
├── notebooks/             # Jupyter notebooks
├── utils/
│   ├── extract_text.py    # Text extraction utilities
│   ├── nlp_processing.py  # NLP processing functions
│   ├── matcher.py         # Matching algorithms
│   └── scoring.py         # Scoring calculations
└── web/
    ├── static/            # Static web assets
    └── templates/         # HTML templates
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.