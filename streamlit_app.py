import streamlit as st
import PyPDF2
import requests
import os
import re
import time
import io

# --- Page Config ---
st.set_page_config(
    page_title="ATS CV Checker",
    page_icon="📄",
    layout="centered"
)

# --- API Configuration ---
# Try different approaches to get the API token
try:
    # First try to get from Streamlit secrets
    HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
except Exception as e:
    # Fallback to environment variable
    HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
    if not HF_API_TOKEN:
        st.error("API Token not found. Please set HF_API_TOKEN in Streamlit secrets or environment variables.")

HF_MODEL_NAME = "google/flan-t5-large"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Utility Functions ---
def clean_text(text):
    """
    Cleans text (e.g., remove extra spaces, special chars).
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
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

    # Limit the size of the input text to avoid API limitations
    cv_text = cv_text[:4000]  # Trim to 4000 chars if longer

    prompt = f"""
    You are an expert in Applicant Tracking Systems (ATS). You provide strict, concise, and accurate ATS compliance analysis.

    Analyze the following CV for ATS compliance:

    CV:
    {cv_text}

    Instructions:

    Follow these steps exactly:

    1.  First, provide an overall ATS compliance score for the CV (0-100).  Output this as: "ATS Compliance Score: [score]" (e.g., "ATS Compliance Score: 75").
    2.  Then, give bullet-point feedback on exactly 3 key areas where the CV can be improved for ATS compliance. Each bullet point MUST be one short, direct sentence. Output this as: "Feedback:\n- [feedback 1]\n- [feedback 2]\n- [feedback 3]"
    3.  Finally, offer 3 specific, actionable suggestions to optimize the CV for ATS. Each suggestion MUST be one short, direct sentence. Output this as: "Suggestions:\n- [suggestion 1]\n- [suggestion 2]\n- [suggestion 3]"

    Output:

    ATS Compliance Score:
    Feedback:
    -
    -
    -
    Suggestions:
    -
    -
    -
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 500,
            "temperature": 0.1,  # Very deterministic
            "do_sample": False,
            "top_p": 0.95,
            "repetition_penalty": 1.1  # Penalize repetition
        }
    }

    for attempt in range(max_retries):
        try:
            with st.spinner(f"Analyzing CV (attempt {attempt + 1}/{max_retries})..."):
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=45,  # Increased timeout
                )
                
                # Handle 503 (model loading) specifically
                if response.status_code == 503:
                    delay = initial_delay * (2 ** attempt)
                    st.info(f"Model is loading. Waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
                
                response.raise_for_status()  # Raise HTTPError for other bad responses

                data = response.json()
                # Better handling of different response formats
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                elif isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                else:
                    st.warning("Unexpected API response format. Raw response:")
                    st.json(data)
                    return "Error: Unexpected API response format."

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"Request timed out. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return "Error: API request timed out after multiple attempts."
                
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"HTTP error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return f"Error: HTTP request failed: {e}"
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"Request error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return f"Error analyzing CV: {e}"
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"Unexpected error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return f"An unexpected error occurred: {e}"

    return "Error: Max retries exceeded. The service may be temporarily unavailable."


def test_api_connection():
    """Test the Hugging Face API connection."""
    test_prompt = "Complete this sentence: The quick brown fox"
    test_payload = {"inputs": test_prompt, "parameters": {"max_length": 20}}

    st.info("Testing API Connection...")
    
    # Display configuration for debugging
    st.write("Configuration:")
    st.write(f"- Model: {HF_MODEL_NAME}")
    st.write(f"- API Token available: {'Yes' if HF_API_TOKEN else 'No'}")

    try:
        with st.spinner("Sending test request..."):
            response = requests.post(
                API_URL,
                headers=headers,
                json=test_payload,
                timeout=15,
            )
        
        st.write(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            st.success("API Connection Test Successful!")
            st.write("Response:")
            st.json(result)
            return True
        elif response.status_code == 503:
            st.warning("Model is loading. Please try again in a few moments.")
            return False
        else:
            st.error(f"API returned error status: {response.status_code}")
            st.write("Response body:")
            st.code(response.text)
            return False
            
    except Exception as e:
        st.error(f"API Connection Test Failed: {e}")
        return False


# --- Streamlit UI ---
st.title("ATS CV Analyzer")
st.markdown(
    """
    Upload your CV (PDF) to get an ATS compliance analysis.
    This tool helps you identify areas for improvement to increase your CV's visibility to Applicant Tracking Systems.
    """
)

# Display configuration tab
with st.expander("🔧 Configuration and Troubleshooting"):
    st.write("Use this section to test your API connection or troubleshoot issues.")
    if st.button("Test API Connection"):
        test_api_connection()

# Main app functionality
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.info("PDF file uploaded. Extracting text...")
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    if extracted_text:
        st.success("Text extracted successfully!")
        
        # Option to view extracted text
        if st.checkbox("View extracted text"):
            st.text_area("Extracted Text", extracted_text, height=200)
        
        # Analyze button
        if st.button("Analyze CV"):
            with st.spinner("Analyzing CV..."):
                analysis_result = analyze_cv_hf(extracted_text)
            
            st.subheader("ATS Analysis Results")
            
            # Format the result for better display
            if "ATS Compliance Score:" in analysis_result:
                # Try to extract and display parts separately for better formatting
                try:
                    # Extract score
                    score_match = re.search(r"ATS Compliance Score:\s*(\d+)", analysis_result)
                    if score_match:
                        score = int(score_match.group(1))
                        st.metric("ATS Compliance Score", f"{score}/100")
                        
                        # Use color based on score
                        if score >= 80:
                            st.success(f"Your CV scored well at {score}%!")
                        elif score >= 60:
                            st.warning(f"Your CV scored {score}% - room for improvement.")
                        else:
                            st.error(f"Your CV scored {score}% - significant improvements needed.")
                    
                    # Extract feedback
                    feedback_section = re.search(r"Feedback:\s*\n(.+?)(?=Suggestions:)", analysis_result, re.DOTALL)
                    if feedback_section:
                        st.subheader("Feedback")
                        feedback_points = re.findall(r"- (.+)", feedback_section.group(1))
                        for point in feedback_points:
                            st.markdown(f"• {point}")
                    
                    # Extract suggestions
                    suggestions_section = re.search(r"Suggestions:\s*\n(.+?)$", analysis_result, re.DOTALL)
                    if suggestions_section:
                        st.subheader("Optimization Suggestions")
                        suggestion_points = re.findall(r"- (.+)", suggestions_section.group(1))
                        for point in suggestion_points:
                            st.markdown(f"• {point}")
                            
                except Exception as e:
                    # If parsing fails, show the raw result
                    st.write(analysis_result)
            else:
                # If the expected format is not found, show the raw result
                st.write(analysis_result)
    else:
        st.error("Failed to extract text from the PDF. Please try another file.")

# Sample test for demonstration
with st.expander("🧪 Run Test Analysis"):
    if st.button("Analyze Sample CV"):
        test_cv_text = """
        John Doe
        Software Engineer
        johndoe@email.com | 555-123-4567 | San Francisco, CA
        
        SUMMARY
        Experienced software engineer with 5 years of experience in web development.
        
        EXPERIENCE
        Senior Developer, Tech Corp
        2018-Present
        - Developed web applications using React and Node.js
        - Implemented CI/CD pipelines
        - Mentored junior developers
        
        Junior Developer, Start-up Inc
        2016-2018
        - Built RESTful APIs
        - Contributed to frontend development
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology, 2016
        
        SKILLS
        JavaScript, React, Node.js, Python, Git, Docker
        """
        
        with st.spinner("Analyzing sample CV..."):
            result = analyze_cv_hf(test_cv_text)
        
        st.subheader("Sample Analysis Result:")
        st.write(result)
