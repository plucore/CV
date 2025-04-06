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
                print(f"Retry attempt {attempt + 2} after {delay} seconds...")
                time.sleep(delay)
            else:
                return f"Error parsing API response: {e}"
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Retry attempt {attempt + 2} after {delay} seconds...")
                time.sleep(delay)
            else:
                return f"An unexpected error occurred: {e}"

    return "Error: Max retries exceeded."


def test_api_connection():
    """
    Temporary function to test the Hugging Face API connection.
    This is for debugging purposes.
    """
    test_prompt = "The quick brown fox jumps over the lazy dog."
    test_payload = {"inputs": test_prompt, "parameters": {"max_length": 50}}

    print("Testing API Connection...")
    print("Model:", HF_MODEL_NAME)
    print("Token Present:", HF_API_TOKEN is not None)  # Check if token is being read

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
            headers=headers,
            json=test_payload,
            timeout=10,
        )
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)  # Log the raw response

        response.raise_for_status()
        result = response.json()
        print("API Response JSON:", result)  # Log the parsed JSON
        st.success(f"API Connection Test Successful: {result}")
    except requests.exceptions.RequestException as e:
        print("RequestException:", e)
        st.error(f"API Connection Test Failed: {e}")
    except Exception as e:
        print("General Exception:", e)
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
if st.button("Run Analyze CV Test"):
    test_cv_text = "John Doe\n123 Main St\nAnytown, CA 91234\nSummary: Highly motivated and results-oriented professional..."  # A short test CV
    result = analyze_cv_hf(test_cv_text)
    st.subheader("Analyze CV Test Result:")
    st.write(result)
