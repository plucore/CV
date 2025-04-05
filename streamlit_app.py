import streamlit as st
import PyPDF2
import requests
import os

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL_NAME = "google/flan-t5-xxl"  # Or your preferred Hugging Face model

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def analyze_cv_hf(cv_text):
    payload = {
        "inputs": f"""
        Analyze the following CV text for ATS compliance.
        Here are the ATS best practices:
        - Use relevant keywords from the job description.
        - Maintain clean and consistent formatting.
        - Structure the CV with clear sections (e.g., Summary, Experience, Skills, Education).
        - Avoid tables or images that may confuse ATS.
        - Use standard section headings.

        CV Text:
        {cv_text}

        Provide:
        1. An ATS compliance score (0-100).
        2. Specific feedback on areas for improvement.
        3. Suggestions for how to improve the CV for ATS.
        """,
        "parameters": {"max_length": 500},
    }
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
            headers=headers,
            json=payload,
        )
        return response.json()[0]["generated_text"]
    except Exception as e:
        st.error(f"Error analyzing CV: {e}")
        return None

st.title("ATS CV Analyzer")
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    extracted_text = extract_text_from_pdf(uploaded_file)
    if extracted_text:
        with st.spinner("Analyzing CV..."):
            analysis_result = analyze_cv_hf(extracted_text)
        if analysis_result:
            st.write(analysis_result)
        else:
            st.error("CV analysis failed.")
