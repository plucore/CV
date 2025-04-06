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
            page_text = page.extract_text()
            if page_text:  # Check if page_text is not None
                text += page_text
        return clean_text(text)  # Clean the extracted text
    except PyPDF2.errors.PdfReadError:
        st.error("Error: Could not read the PDF. It might be corrupted or not a valid PDF.")
        return None
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None


def analyze_cv_hf(cv_text, max_retries=3, initial_delay=1):
    """Analyzes CV text using Hugging Face Inference API."""
    if not cv_text:
        return "Error: No CV text to analyze."  # Handle empty input

    prompt = "Analyze the following CV for ATS compliance:\n\n"  # Explicit newlines
    prompt += "CV:\n"
    prompt += f"{cv_text}\n\n"  # Insert CV text
    prompt += "Instructions:\n\n"
    prompt += "1. Provide an overall ATS compliance score (0-100).\n"
    prompt += "2. Give bullet-point feedback on 3 key areas for improvement.\n"
    prompt += "3. Offer 3 specific, actionable suggestions to optimize the CV for ATS.\n\n"
    prompt += "Output:\n"

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 400}  # Adjust as needed
    }

    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Retry attempt {attempt + 2} after {delay} seconds...")
                time.sleep(delay)
            else:
                return "Error: API request timed out."
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Retry attempt {attempt + 2} after {delay} seconds...")
                time.sleep(delay)
            else:
                return f"Error analyzing CV: {e}"
        except ValueError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Retry attempt {attempt +
