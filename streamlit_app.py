import streamlit as st
import PyPDF2
import requests
import os
import re  # For potential regex-based cleaning
import time  # For potential retry logic

# --- API Configuration ---
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN")  # Streamlit Secrets (most secure)
HF_MODEL_NAME = "google/flan-t5-large"  # Changed to flan-t5-large
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

    prompt = f"""
    Analyze the following CV for ATS compliance:

    CV:
    {cv_text}

    Instructions:

    1.  Provide an overall ATS compliance score (0-100).
    2.  Give bullet-point feedback on 3 key areas for improvement.
    3.  Offer 3 specific, actionable suggestions to optimize the CV for ATS.

    Output:
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 400}  # Adjust as needed
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


def test_api_connection():
    """
    Temporary function to test the Hugging Face API connection.
    This is for debugging purposes.
    """
    test_prompt = "The quick brown fox jumps over the lazy dog."
    test_payload = {"inputs": test_prompt, "parameters": {"max_length": 50}}

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
            headers=headers,
            json=test_payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        st.success(f"API Connection Test Successful: {result}")
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Test Failed: {e}")
    except Exception as e:
        st.error(f"API Test Error: {e}")


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

# --- Temporary Test Button ---
if st.button("Test API Connection"):
    test_api_connection()
