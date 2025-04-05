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
    print("analyze_cv_hf function called")
    print("Payload:", payload)

    try:
        print("Sending request to Hugging Face API...")
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
            headers=headers,
            json=payload,
        )
        print("API Response (status code):", response.status_code)
        print("API Response (text):", response.text)

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        print("API Response (JSON):", data)

        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            result = data[0]["generated_text"]
            print("Generated text found:", result)
            return result
        else:
            print("Unexpected API response structure")
            return None

    except requests.exceptions.RequestException as e:
        print("RequestException:", e)
        st.error(f"Error analyzing CV: {e}")
        return None
    except ValueError as e:
        print("ValueError:", e)
        st.error(f"Error parsing API response: {e}")
        return None
    except Exception as e:
        print("Exception:", e)
        st.error(f"An unexpected error occurred: {e}")
        return None
