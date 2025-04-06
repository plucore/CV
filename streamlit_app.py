import streamlit as st
import PyPDF2
import requests
import os
import re  # For potential regex-based cleaning

# --- API Configuration ---
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN")  # Streamlit Secrets (most secure)
HF_MODEL_NAME = "google/flan-t5-large"  # Or your preferred Hugging Face model
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


# --- Utility Functions ---
def clean_text(text):
    """
    Optional: Cleans text (e.g., remove extra spaces, special chars).
    You might want to customize this further based on your needs.
    """
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle empty pages
        return clean_text(text)  # Clean the extracted text
    except PyPDF2.errors.PdfReadError:
        st.error("Error: Could not read the PDF. It might be corrupted or not a valid PDF.")
        return None
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None


def analyze_cv_hf(cv_text):
    """Analyzes CV text using Hugging Face Inference API."""
    if not cv_text:
        return "Error: No CV text to analyze."  # Handle empty input

    payload = {
        "inputs": f"Analyze the following CV text for ATS compliance. Here are the ATS best practices: "
                  f"- Use relevant keywords from the job description. "
                  f"- Maintain clean and consistent formatting. "
                  f"- Structure the CV with clear sections (e.g., Summary, Experience, Skills, Education). "
                  f"- Avoid tables or images that may confuse ATS. "
                  f"- Use standard section headings. CV Text: {cv_text} Provide: "
                  f"1. An ATS compliance score (0-100). "
                  f"2. Specific feedback on areas for improvement. "
                  f"3. Suggestions for how to improve the CV for ATS.",
        "parameters": {"max_length": 500},  # Adjust as needed
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=30,  # Add timeout (in seconds)
        )
        response.raise_for_status()  # Raise HTTPError for bad responses

        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return clean_text(data[0]["generated_text"])
        else:
            return "Error: Unexpected API response."

    except requests.exceptions.Timeout:
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error analyzing CV: {e}"
    except ValueError as e:
        return f"Error parsing API response: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- Streamlit UI ---
st.title("ATS CV Analyzer")
st.markdown(
    """
    Upload your CV (PDF) to get an ATS compliance analysis.
    This tool helps you identify areas for improvement to increase your CV's visibility to Applicant Tracking Systems.
    """
)

uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    extracted_text = extract_text_from_pdf(uploaded_file)
    if extracted_text:
        with st.spinner("Analyzing CV..."):
            analysis_result = analyze_cv_hf(extracted_text)
        st.subheader("Analysis Results")
        st.write(analysis_result)
