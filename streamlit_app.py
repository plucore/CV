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

    print("analyze_cv_hf called with cv_text:", cv_text)  # Log input

    prompt = f"""
    Analyze the following CV for ATS compliance:

    CV:
    {cv_text}

    Instructions:

    1.  Provide an overall ATS compliance score (0-100).
    2.  Give bullet-point feedback on
