import streamlit as st
import joblib
import numpy as np
import re
import urllib.parse as urlparse
import time
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                  ShadCN Neutral Theme — CSS                     ║
# ╚══════════════════════════════════════════════════════════════════╝
THEME_CSS = """
<style>
    /* ===== Google Fonts ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ===== ShadCN Neutral Design Tokens ===== */
    :root {
        --background: #ffffff;
        --foreground: #0a0a0a;
        --card: #ffffff;
        --card-foreground: #0a0a0a;
        --primary: #171717;
        --primary-foreground: #fafafa;
        --secondary: #f5f5f5;
        --secondary-foreground: #171717;
        --muted: #f5f5f5;
        --muted-foreground: #737373;
        --accent: #f5f5f5;
        --accent-foreground: #171717;
        --destructive: #ef4444;
        --border: #e5e5e5;
        --input: #e5e5e5;
        --ring: #0a0a0a;
        --radius: 0.5rem;
        --success: #16a34a;
        --success-bg: #f0fdf4;
        --success-border: #bbf7d0;
        --danger: #dc2626;
        --danger-bg: #fef2f2;
        --danger-border: #fecaca;
        --warning-bg: #fffbeb;
        --warning-border: #fde68a;
        --warning: #d97706;
    }

    /* ===== Global ===== */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background-color: var(--background) !important;
        color: var(--foreground) !important;
    }

    .main .block-container {
        padding: 2rem 1rem 2rem 1rem !important;
        max-width: 760px !important;
    }

    /* ===== Typography ===== */
    h1, h2, h3, h4, h5, h6, p, label, div, li, td, th, input, button, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Restore Material Symbols font for Streamlit icons */
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }

    h1 { font-weight: 700 !important; letter-spacing: -0.025em !important; color: var(--foreground) !important; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.015em !important; color: var(--foreground) !important; }

    /* ===== Hide Streamlit Branding (keep header for sidebar toggle) ===== */
    #MainMenu, footer { visibility: hidden !important; }
    [data-testid="stHeader"] { background-color: transparent !important; }

    /* ===== Divider ===== */
    hr { border-color: var(--border) !important; opacity: 0.5 !important; }

    /* ─────────────────────────────────────────────
       SIDEBAR TOGGLE (Collapse & Expand buttons)
    ───────────────────────────────────────────── */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stExpandSidebarButton"] button {
        color: #0a0a0a !important;
    }

    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
        color: #0a0a0a !important;
    }

    [data-testid="stExpandSidebarButton"] {
        background-color: var(--background) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.08) !important;
    }

    [data-testid="stSidebar"] {
        background-color: #fafafa !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.9375rem !important;
        font-weight: 600 !important;
        color: var(--foreground) !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--muted-foreground) !important;
        font-size: 0.75rem !important;
    }

    /* ─────────────────────────────────────────────
       TEXT INPUT
    ───────────────────────────────────────────── */
    [data-testid="stTextInput"] input {
        border: 1px solid var(--input) !important;
        border-radius: var(--radius) !important;
        padding: 0.625rem 0.875rem !important;
        font-size: 0.875rem !important;
        background-color: var(--background) !important;
        color: var(--foreground) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        height: auto !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: var(--ring) !important;
        box-shadow: 0 0 0 3px rgba(10, 10, 10, 0.06) !important;
        outline: none !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: var(--muted-foreground) !important;
    }

    [data-testid="stTextInput"] label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--foreground) !important;
    }

    /* ─────────────────────────────────────────────
       PRIMARY BUTTON
    ───────────────────────────────────────────── */
    .stButton > button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: var(--primary-foreground) !important;
        border: 1px solid var(--primary) !important;
        border-radius: var(--radius) !important;
        padding: 0.5625rem 1.25rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        transition: opacity 0.15s ease !important;
        letter-spacing: 0 !important;
    }

    .stButton > button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        opacity: 0.88 !important;
    }

    .stButton > button[data-testid="baseButton-primary"]:active,
    .stButton > button[kind="primary"]:active {
        transform: scale(0.98) !important;
    }

    /* ─────────────────────────────────────────────
       METRIC CARDS
    ───────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.75rem 0.875rem !important;
    }

    [data-testid="stMetric"] label {
        color: var(--muted-foreground) !important;
        font-size: 0.6875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.375rem !important;
        font-weight: 700 !important;
        color: var(--foreground) !important;
    }

    /* ─────────────────────────────────────────────
       ALERTS
    ───────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-size: 0.875rem !important;
        border: 1px solid var(--border) !important;
    }

    /* ─────────────────────────────────────────────
       DATAFRAME / TABLE
    ───────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
    }

    /* ─────────────────────────────────────────────
       BAR CHART
    ───────────────────────────────────────────── */
    [data-testid="stVegaLiteChart"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.5rem !important;
        background-color: var(--card) !important;
    }

    /* ─────────────────────────────────────────────
       BORDERED CONTAINER (st.container(border=True))
    ───────────────────────────────────────────── */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--border) !important;
        border-radius: 0.75rem !important;
        background-color: var(--card) !important;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.04) !important;
    }

    /* ═══════════════════════════════════════════
       CUSTOM HTML COMPONENTS
    ═══════════════════════════════════════════ */

    /* --- Hero Banner --- */
    .hero-banner {
        background: linear-gradient(145deg, #fafafa 0%, #f0f0f0 100%);
        border: 1px solid var(--border);
        border-radius: 0.75rem;
        padding: 2rem 2rem 1.75rem 2rem;
        margin-bottom: 1.5rem;
    }

    .hero-banner .hero-icon {
        font-size: 2.25rem;
        margin-bottom: 0.5rem;
        line-height: 1;
    }

    .hero-banner h1 {
        font-size: 1.625rem !important;
        margin: 0 0 0.375rem 0 !important;
        line-height: 1.3 !important;
    }

    .hero-banner .hero-sub {
        color: var(--muted-foreground);
        font-size: 0.9rem;
        line-height: 1.65;
        margin: 0;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        background: var(--secondary);
        color: var(--secondary-foreground);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 500;
        border: 1px solid var(--border);
        margin-top: 1rem;
    }

    .steps-list {
        list-style: none;
        padding: 0;
        margin: 0.875rem 0 0 0;
    }

    .steps-list li {
        font-size: 0.8125rem;
        color: var(--muted-foreground);
        padding: 0.3rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.25rem;
        height: 1.25rem;
        border-radius: 50%;
        background: var(--secondary);
        border: 1px solid var(--border);
        font-size: 0.6875rem;
        font-weight: 600;
        color: var(--foreground);
        flex-shrink: 0;
    }

    /* --- Section Label --- */
    .section-label {
        font-size: 0.6875rem;
        font-weight: 600;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }

    /* --- Result Cards --- */
    .result-card {
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.875rem;
    }

    .result-card.safe {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-left: 4px solid var(--success);
    }

    .result-card.danger {
        background: var(--danger-bg);
        border: 1px solid var(--danger-border);
        border-left: 4px solid var(--danger);
    }

    .result-card.caution {
        background: var(--warning-bg);
        border: 1px solid var(--warning-border);
        border-left: 4px solid var(--warning);
    }

    .result-card .r-icon {
        font-size: 1.375rem;
        flex-shrink: 0;
        line-height: 1.2;
    }

    .result-card .r-body h4 {
        margin: 0 0 0.25rem 0 !important;
        font-size: 0.9375rem !important;
        font-weight: 600 !important;
    }

    .result-card.safe .r-body h4 { color: #15803d !important; }
    .result-card.danger .r-body h4 { color: #b91c1c !important; }
    .result-card.caution .r-body h4 { color: #92400e !important; }

    .result-card .r-body p {
        margin: 0 !important;
        font-size: 0.8125rem !important;
        color: var(--muted-foreground) !important;
        line-height: 1.55 !important;
    }

    /* --- Footer --- */
    .app-footer {
        text-align: center;
        padding: 1.25rem 0 0.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid var(--border);
        font-size: 0.75rem;
        color: var(--muted-foreground);
        line-height: 1.8;
    }

    .app-footer .ver-badge {
        background: var(--secondary);
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
        font-weight: 500;
        border: 1px solid var(--border);
        font-size: 0.6875rem;
    }

    /* --- Sidebar Section Cards --- */
    .sidebar-section {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-label {
        font-size: 0.6875rem;
        font-weight: 600;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       Asset Caching                            ║
# ╚══════════════════════════════════════════════════════════════════╝
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  Session State — Analytics                      ║
# ╚══════════════════════════════════════════════════════════════════╝
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "phishing_count" not in st.session_state:
    st.session_state.phishing_count = 0
if "safe_count" not in st.session_state:
    st.session_state.safe_count = 0

def record_scan(url, result_label):
    """Records a scan result into session state analytics."""
    st.session_state.total_scans += 1
    if result_label == "Phishing":
        st.session_state.phishing_count += 1
    else:
        st.session_state.safe_count += 1
    st.session_state.scan_history.append({
        "#": st.session_state.total_scans,
        "URL": url if len(url) <= 60 else url[:57] + "...",
        "Result": result_label,
        "Time": datetime.now().strftime("%I:%M:%S %p"),
    })


# ╔══════════════════════════════════════════════════════════════════╗
# ║              Feature Extraction Logic (30 Features)             ║
# ╚══════════════════════════════════════════════════════════════════╝
def extract_features(url):
    """
    Extracts 30 features from a given URL to match the model's training data.
    - Features that can be calculated from the URL are implemented.
    - Features requiring external services now use a neutral placeholder (0).
    """
    features = []

    # 1. having_IP_Address
    try:
        domain = urlparse.urlparse(url).hostname
        features.append(1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else -1)
    except:
        features.append(1)

    # 2. URL_Length
    features.append(1 if len(url) >= 54 else -1)

    # 3. Shortining_Service
    features.append(1 if any(service in url for service in ['bit.ly', 'goo.gl', 't.co', 'tinyurl']) else -1)

    # 4. having_At_Symbol
    features.append(1 if '@' in url else -1)

    # 5. double_slash_redirecting
    features.append(1 if url.rfind('//') > 7 else -1)

    # 6. Prefix_Suffix
    features.append(1 if '-' in urlparse.urlparse(url).netloc else -1)

    # 7. having_Sub_Domain
    features.append(1 if urlparse.urlparse(url).netloc.count('.') > 2 else -1)

    # 8. SSLfinal_State
    features.append(-1 if url.startswith('https') else 1)

    # 9-30: Placeholders for features requiring external services
    features.extend([0] * 22)

    return np.array(features).reshape(1, -1)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        MAIN UI LAYOUT                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# --- Hero Banner ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">🛡️</div>
    <h1>Phishing Website Detector</h1>
    <p class="hero-sub">
        Enterprise-grade URL threat analysis powered by machine learning.
        Identify and flag potentially malicious websites in real-time.
    </p>
    <ul class="steps-list">
        <li><span class="step-num">1</span> Enter a full website URL below</li>
        <li><span class="step-num">2</span> Click <strong>Analyze URL</strong> to run detection</li>
        <li><span class="step-num">3</span> Review the prediction result</li>
    </ul>
    <div class="hero-badge">
        <span>🤖</span> Random Forest Classifier · 30 URL Features
    </div>
</div>
""", unsafe_allow_html=True)

# --- URL Scanner Section ---
if model is None:
    st.error("🚨 **Model Not Found!**")
    st.warning(
        "The model file `phishing_model.pkl` is missing. "
        "Please run the `train.py` script in your terminal to create it.",
        icon="⚙️"
    )
else:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">🔍 URL Scanner</div>',
            unsafe_allow_html=True
        )
        url_input = st.text_input(
            "Target URL",
            placeholder="e.g., https://example.com or http://192.168.1.1/login",
            label_visibility="collapsed"
        )
        analyze_clicked = st.button(
            "Analyze URL", type="primary", use_container_width=True
        )

    # --- Prediction Logic ---
    if analyze_clicked:
        if url_input and url_input.strip() != "":
            # Prepend http if scheme is missing for parsing
            if not url_input.startswith(('http://', 'https://')):
                url_input = 'http://' + url_input

            try:
                with st.spinner('Analyzing URL... This may take a moment.'):
                    time.sleep(1)
                    features = extract_features(url_input)
                    prediction = model.predict(features)

                st.markdown(
                    '<div class="section-label" style="margin-top:1rem;">📋 Analysis Result</div>',
                    unsafe_allow_html=True
                )

                # The model predicts -1 for legitimate and 1 for phishing
                if prediction[0] == -1:
                    st.markdown("""
                    <div class="result-card safe">
                        <div class="r-icon">✅</div>
                        <div class="r-body">
                            <h4>Legitimate — No Threat Detected</h4>
                            <p>This URL appears to be safe. No phishing indicators were found
                            in the URL structure analysis.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    record_scan(url_input, "Legitimate")
                else:
                    st.markdown("""
                    <div class="result-card danger">
                        <div class="r-icon">🚨</div>
                        <div class="r-body">
                            <h4>Phishing — Threat Detected</h4>
                            <p>This URL shows characteristics commonly associated with phishing attempts.
                            Do not enter personal or financial information.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="result-card caution">
                        <div class="r-icon">⚠️</div>
                        <div class="r-body">
                            <h4>Security Advisory</h4>
                            <p>Exercise extreme caution with this URL. If you've already visited
                            it, consider changing any passwords you may have entered.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    record_scan(url_input, "Phishing")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
        else:
            st.warning("Please enter a URL to analyze.", icon="❗")

# --- Footer ---
st.markdown("""
<div class="app-footer">
    <span class="ver-badge">v1.0.0</span><br>
    Phishing Website Detector · Built with Streamlit & Scikit-learn<br>
    <em>This tool is for informational purposes only. Always verify with additional sources.</em>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   SIDEBAR — Analytics Dashboard                 ║
# ╚══════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown(
        '<div class="sidebar-label" style="margin-bottom:0.25rem;">📊 Analytics Dashboard</div>',
        unsafe_allow_html=True
    )
    st.caption("Real-time session statistics")
    st.write("---")

    # --- Key Metrics ---
    col1, col2 = st.columns(2)
    col1.metric("Total Scans", st.session_state.total_scans)
    col2.metric(
        "Threat Rate",
        f"{(st.session_state.phishing_count / st.session_state.total_scans * 100):.0f}%"
        if st.session_state.total_scans > 0 else "—",
    )

    col3, col4 = st.columns(2)
    col3.metric("✅ Safe", st.session_state.safe_count)
    col4.metric("🚨 Phishing", st.session_state.phishing_count)

    st.write("---")

    # --- Chart ---
    st.markdown(
        '<div class="sidebar-label">Result Breakdown</div>',
        unsafe_allow_html=True
    )
    if st.session_state.total_scans > 0:
        # Create DataFrame without a custom index so it defaults to 0, 1, ...
        # and explicitly use columns "Category" and "Count" for cleaner table viewing
        chart_data = pd.DataFrame({
            "Category": ["Legitimate", "Phishing"],
            "Count": [st.session_state.safe_count, st.session_state.phishing_count]
        })
        st.bar_chart(chart_data, x="Category", y="Count", color=["#2ecc71"])
    else:
        st.info("Scan some URLs to see chart data.", icon="💡")

    st.write("---")

    # --- Scan History ---
    st.markdown(
        '<div class="sidebar-label">🕰️ Scan History</div>',
        unsafe_allow_html=True
    )
    if st.session_state.scan_history:
        history_df = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No scans yet. Analyze a URL to begin.")
