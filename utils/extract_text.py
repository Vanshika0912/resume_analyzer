import pdfplumber
import docx2txt

# ----------------------------------------------
# Extract text from PDF
# ----------------------------------------------
def extract_text_from_pdf(file):
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print("PDF extraction error:", e)
        return ""

# ----------------------------------------------
# Extract text from DOCX
# ----------------------------------------------
def extract_text_from_docx(file):
    try:
        return docx2txt.process(file)
    except Exception as e:
        print("DOCX extraction error:", e)
        return ""
