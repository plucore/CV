import streamlit as st
import PyPDF2
import requests
import re
import time
import io

# Set page config
st.set_page_config(page_title="ATS CV Analyzer", page_icon="📄")

# API Configuration - Using direct access to secrets
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
HF_MODEL_NAME = "google/flan-t5-large"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        file.seek(0)  # Reset file pointer
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return re.sub(r'\s+', ' ', text).strip()  # Clean the text
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return None

def analyze_cv(cv_text):
    """Analyze CV text using the HF API."""
    if not cv_text:
        return "Error: No text to analyze."
    
    # Trim text if too long
    cv_text = cv_text[:3500]
    
    prompt = f"""
    You are an expert in Applicant Tracking Systems (ATS). Analyze the following CV for ATS compliance:

    CV:
    {cv_text}

    Instructions:
    1. First, provide an overall ATS compliance score for the CV (0-100). Output: "ATS Compliance Score: [score]"
    2. Then, give bullet-point feedback on exactly 3 key areas where the CV can be improved for ATS compliance. Output: "Feedback:\n- [feedback 1]\n- [feedback 2]\n- [feedback 3]"
    3. Finally, offer 3 specific, actionable suggestions to optimize the CV for ATS. Output: "Suggestions:\n- [suggestion 1]\n- [suggestion 2]\n- [suggestion 3]"
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 500,
            "temperature": 0.1,
            "do_sample": False,
        }
    }

    # Try up to 3 times with exponential backoff
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            # If model is loading
            if response.status_code == 503:
                wait_time = 2 ** attempt
                st.warning(f"Model is loading. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            # If other error
            if response.status_code != 200:
                return f"Error: API returned status code {response.status_code}"
                
            # Success - parse response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Error: Could not parse response")
            else:
                return "Error: Unexpected response format"
                
        except Exception as e:
            if attempt < 2:  # Try again if not last attempt
                time.sleep(2 ** attempt)
            else:
                return f"Error: {str(e)}"
    
    return "Error: Failed after multiple attempts"

# UI Components
st.title("ATS CV Analyzer")
st.write("Upload your CV to check its ATS compliance")

# Upload section
uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=["pdf"])

if uploaded_file:
    # Extract text button
    if st.button("Analyze CV"):
        # Extract text
        with st.spinner("Extracting text from PDF..."):
            cv_text = extract_text_from_pdf(uploaded_file)
            
        if cv_text:
            st.success("Text extracted successfully!")
            
            # Analyze the CV
            with st.spinner("Analyzing CV with AI..."):
                analysis = analyze_cv(cv_text)
            
            # Display results
            st.subheader("Analysis Results")
            st.write(analysis)
        else:
            st.error("Could not extract text from the PDF. Please try a different file.")
            
    # Option to view extracted text
    if st.checkbox("View extracted text"):
        text = extract_text_from_pdf(uploaded_file)
        if text:
            st.text_area("Extracted text", text, height=200)
        else:
            st.error("Could not extract text from the PDF.")

# Add a quick test option
if st.checkbox("Run test analysis"):
    sample_text = """
    John Doe
    Software Engineer
    123 Main St, City, State
    Phone: 555-123-4567
    Email: john@example.com
    
    EXPERIENCE
    Senior Developer, XYZ Corp (2018-Present)
    - Led development of company's flagship product
    - Managed team of 5 junior developers
    
    Developer, ABC Inc (2015-2018)
    - Developed web applications using React
    
    EDUCATION
    BS Computer Science, State University (2015)
    
    SKILLS
    Programming: JavaScript, Python, Java
    Tools: Git, Docker, AWS
    """
    
    if st.button("Test Analysis"):
        with st.spinner("Running test analysis..."):
            test_result = analyze_cv(sample_text)
        st.subheader("Test Analysis Result")
        st.write(test_result)
