import streamlit as st
import joblib
import numpy as np
import re
import urllib.parse as urlparse
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Asset Caching ---
# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_model():
    try:
        model = joblib.load("phishing_model.pkl")
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None

model = load_model()

# --- Feature Extraction Logic (30 Features) ---
def extract_features(url):
    """
    Extracts 30 features from a given URL to match the model's training data.
    - Features that can be calculated from the URL are implemented.
    - Features requiring external services now use a neutral placeholder (0).
    """
    features = []

    # 1. having_IP_Address: 1 for phishing, -1 for legitimate.
    try:
        domain = urlparse.urlparse(url).hostname
        features.append(1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else -1)
    except:
        features.append(1)

    # 2. URL_Length: 1 for phishing (long), -1 for legitimate (short).
    features.append(1 if len(url) >= 54 else -1)

    # 3. Shortining_Service: 1 for phishing, -1 for legitimate.
    features.append(1 if any(service in url for service in ['bit.ly', 'goo.gl', 't.co', 'tinyurl']) else -1)

    # 4. having_At_Symbol: 1 for phishing, -1 for legitimate.
    features.append(1 if '@' in url else -1)

    # 5. double_slash_redirecting: 1 for phishing, -1 for legitimate.
    features.append(1 if url.rfind('//') > 7 else -1)

    # 6. Prefix_Suffix: 1 for phishing, -1 for legitimate.
    features.append(1 if '-' in urlparse.urlparse(url).netloc else -1)

    # 7. having_Sub_Domain: 1 for phishing (multiple subdomains), -1 for legitimate.
    # We check for more than 2 dots, which is a common phishing tactic.
    features.append(1 if urlparse.urlparse(url).netloc.count('.') > 2 else -1)

    # 8. SSLfinal_State: 1 for phishing (no HTTPS), -1 for legitimate (HTTPS).
    features.append(-1 if url.startswith('https') else 1)

    # --- THE FIX IS HERE: Using a neutral placeholder (0) ---
    # Append placeholders for features 9-30 that we cannot calculate live.
    features.extend([0] * 22)

    return np.array(features).reshape(1, -1)


# --- UI Layout ---
st.title("🛡️ Phishing Website Detector")
st.markdown(
    """
    Welcome to the Phishing Website Detector! This tool uses a machine learning model
    to help you identify potentially malicious websites.
    
    **How to use:**
    1.  Enter a full website URL in the text box below.
    2.  Click the "Analyze URL" button.
    3.  Review the prediction result.
    """
)
st.write("---")

# --- User Input and Prediction ---
if model is None:
    st.error("🚨 **Model Not Found!**")
    st.warning("The model file `phishing_model.pkl` is missing. Please run the `train.py` script in your terminal to create it.", icon="⚙️")
else:
    url_input = st.text_input(
        "Enter a website URL to analyze:",
        placeholder="e.g., http://121.18.238.11/login.html"
    )

    if st.button("Analyze URL", type="primary"):
        if url_input and url_input.strip() != "":
            # Prepend http if scheme is missing for parsing
            if not url_input.startswith(('http://', 'https://')):
                url_input = 'http://' + url_input

            try:
                with st.spinner('Analyzing URL... This may take a moment.'):
                    time.sleep(1) # Simulate analysis time
                    features = extract_features(url_input)
                    prediction = model.predict(features)
                    
                st.write("---")
                st.subheader("Analysis Result")

                # NOTE: The model predicts -1 for legitimate and 1 for phishing
                if prediction[0] == -1:
                    st.success("✅ This URL appears to be **Legitimate**.", icon="👍")
                    st.balloons()
                else:
                    st.error("⚠️ This URL is likely a **Phishing** attempt.", icon="🚨")
                    st.warning("Please be cautious. Avoid entering any personal or financial information on this site.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
        else:
            st.warning("Please enter a URL to analyze.", icon="❗")

