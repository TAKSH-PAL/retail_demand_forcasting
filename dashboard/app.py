import streamlit as st
import requests
import datetime
import pandas as pd
import time

# Set page config for dynamic layout and browser title
st.set_page_config(
    page_title="Rossmann Demand Forecaster",
    page_icon=":material/trending_up:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Geist and Geist Mono, override native styles with Vercel UI theme tokens
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=geist@400,500,600,700&f[]=geist-mono@400,500&display=swap');

    /* Global style overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Geist', -apple-system, sans-serif !important;
        background-color: #fafafa !important;
        color: #171717 !important;
    }

    /* Custom Title Style matching display-xl */
    .vercel-title {
        font-family: 'Geist', sans-serif !important;
        font-size: 3rem !important;
        font-weight: 600 !important;
        letter-spacing: -2.4px !important;
        color: #171717 !important;
        line-height: 48px !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Subtitle matching body-lg */
    .vercel-subtitle {
        font-family: 'Geist', sans-serif !important;
        font-size: 1.125rem !important;
        font-weight: 400 !important;
        line-height: 28px !important;
        color: #4d4d4d !important;
        margin-bottom: 2rem;
    }

    /* Section Headings matching display-lg */
    .vercel-heading {
        font-family: 'Geist', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.96px !important;
        color: #171717 !important;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* In-app console labels matching caption-mono */
    .caption-mono {
        font-family: 'Geist Mono', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        color: #888888 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0.5rem;
    }

    /* Inset Hairline Box Styling with Stacked level-3 shadows */
    .st-emotion-cache-1r6g8gg, .st-emotion-cache-vk38na, div[data-testid="stVerticalBlock"] > div[style*="border:"] {
        border: 1px solid #ebebeb !important;
        border-radius: 8px !important;
        box-shadow: 0px 2px 2px rgba(0,0,0,0.02), 0px 8px 8px -8px rgba(0,0,0,0.04) !important;
        background-color: #ffffff !important;
        padding: 1.5rem !important;
    }

    /* Linear Mesh Gradient Top Divider Bar */
    .gradient-bar {
        background: linear-gradient(90deg, #007cf0 0%, #00dfd8 33%, #7928ca 66%, #ff0080 100%);
        height: 6px;
        width: 100%;
        border-radius: 3px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Top Decorative Brand Bar
st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

# 2. Sidebar Configuration
st.sidebar.markdown('<div class="caption-mono">Console Configuration</div>', unsafe_allow_html=True)
api_url_input = st.sidebar.text_input(
    "Endpoint URL",
    value="https://retail-forecast-service.onrender.com",
    help="FastAPI microservice endpoint URL"
)

st.sidebar.markdown("---")

# Render Wakeup polling loop to solve Render inactivity spin downs
if 'connected_endpoints' not in st.session_state:
    st.session_state['connected_endpoints'] = set()

if api_url_input not in st.session_state['connected_endpoints']:
    status_placeholder = st.sidebar.empty()
    progress_placeholder = st.sidebar.empty()
    
    with status_placeholder.container():
        st.info("😴 Waking up Render backend container. Please stand by (takes 50-90s on free tier)...")
        
    max_retries = 24  # 120s max
    woke_up = False
    for i in range(max_retries):
        progress_val = min(1.0, (i + 1) / max_retries)
        progress_placeholder.progress(progress_val)
        try:
            r = requests.get(f"{api_url_input}/", timeout=3)
            if r.status_code == 200:
                woke_up = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
        
    status_placeholder.empty()
    progress_placeholder.empty()
    
    if woke_up:
        st.session_state['connected_endpoints'].add(api_url_input)
        st.sidebar.success("🟢 Connection established!")
    else:
        st.sidebar.error("❌ Connection timed out. Refresh page to try again.")

@st.cache_data(ttl=60)
def fetch_model_metadata(base_url):
    try:
        r = requests.get(f"{base_url}/model", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

model_info = fetch_model_metadata(api_url_input)

if model_info:
    st.sidebar.markdown(f"""
    <div style="background-color: #fafafa; border: 1px solid #ebebeb; padding: 1rem; border-radius: 6px;">
        <div class="caption-mono" style="margin-bottom: 0.25rem;">Active Service</div>
        <p style="font-size: 1.1rem; font-weight: 500; margin: 0 0 0.5rem 0;">{model_info['name']}</p>
        <div class="caption-mono" style="margin-bottom: 0.25rem;">Validation Performance</div>
        <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">R² Score: <b>{model_info['r2']:.4f}</b><br>RMSE: <b>{model_info['rmse']}</b></p>
        <div class="caption-mono" style="margin-bottom: 0.25rem;">Compiled Date</div>
        <p style="margin: 0; font-size: 0.9rem;">{model_info['trained']}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning("⚠️ Connection to deployed service could not be established. Check console configuration.")

# 3. Main Hero Headers
st.markdown('<div class="vercel-title">Retail Demand forecasting.</div>', unsafe_allow_html=True)
st.markdown('<div class="vercel-subtitle">Enter parameters below to extract stationary lag and rolling statistics dynamically for real-time demand inference.</div>', unsafe_allow_html=True)

# 4. Form Parameter Inputs
with st.container(border=True):
    st.markdown('<div class="caption-mono">Input Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        store_id = st.number_input(
            "Store Index ID",
            min_value=1,
            max_value=1115,
            value=15,
            help="Unique Store code (1-1115)"
        )
        
        forecast_date = st.date_input(
            "Transaction Date",
            value=datetime.date(2015, 8, 3),
            min_value=datetime.date(2013, 1, 1),
            help="Pick transaction date"
        )

    with col2:
        promo_active = st.checkbox(
            "Promotion Active (Promo)",
            value=True,
            help="Toggle active store promotion campaign"
        )
        
        state_holiday_label = st.segmented_control(
            "State Holiday classification",
            options=["None", "Public Holiday (a)", "Easter (b)", "Christmas (c)"],
            default="None",
            help="Specify holiday classification"
        )
        
        school_holiday_active = st.checkbox(
            "School Holiday active",
            value=False,
            help="Toggle active school holidays"
        )

# Map labels to codes
holiday_map = {
    "None": "0",
    "Public Holiday (a)": "a",
    "Easter (b)": "b",
    "Christmas (c)": "c"
}
state_holiday = holiday_map[state_holiday_label or "None"]

# Action Button
st.markdown("<br>", unsafe_allow_html=True)
trigger_forecast = st.button("Generate Forecast", width="stretch", icon=":material/trending_up:")

if trigger_forecast:
    payload = {
        "store": int(store_id),
        "date": forecast_date.strftime("%Y-%m-%d"),
        "promo": 1 if promo_active else 0,
        "state_holiday": state_holiday,
        "school_holiday": 1 if school_holiday_active else 0
    }
    
    with st.spinner("Dispatching query to live service..."):
        try:
            response = requests.post(f"{api_url_input}/predict", json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown('<div class="vercel-heading">Forecast results.</div>', unsafe_allow_html=True)
                
                # Dynamic visual blocks
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    with st.container(border=True):
                        st.markdown('<div class="caption-mono">Forecast Turnover</div>', unsafe_allow_html=True)
                        st.markdown(f"<h2 style='margin: 0; color: #171717;'>€{result['predicted_sales']:,.2f}</h2>", unsafe_allow_html=True)
                        
                with m_col2:
                    with st.container(border=True):
                        st.markdown('<div class="caption-mono">Confidence Level</div>', unsafe_allow_html=True)
                        conf = result['confidence']
                        badge_color = "#d3e5ff" if conf == "High" else "#ffefcf"
                        text_color = "#0070f3" if conf == "High" else "#ab570a"
                        st.markdown(f"""
                        <div style="background-color: {badge_color}; color: {text_color}; display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 500; font-size: 1.1rem; margin-top: 0.5rem;">
                            {conf}
                        </div>
                        """, unsafe_allow_html=True)
                        
                with m_col3:
                    with st.container(border=True):
                        st.markdown('<div class="caption-mono">Expected Demand</div>', unsafe_allow_html=True)
                        demand = result['demand_level']
                        badge_color = "#d3e5ff" if demand == "High" else "#e2e8f0" if demand == "Normal" else "#f7d4d6"
                        text_color = "#0070f3" if demand == "High" else "#4d4d4d" if demand == "Normal" else "#ee0000"
                        st.markdown(f"""
                        <div style="background-color: {badge_color}; color: {text_color}; display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 500; font-size: 1.1rem; margin-top: 0.5rem;">
                            {demand}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recommendation banner
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown('<div class="caption-mono">Operations Directive</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 1.1rem; margin: 0.5rem 0 0 0;'>{result['inventory_recommendation']}</p>", unsafe_allow_html=True)
                    
                # Tech expansion details using monospaced styling
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 Trace Pipeline Execution Details"):
                    st.markdown(f"""
                    ```text
                    [1/7] Fetching Store {store_id} transaction logs (2015-05-15 onwards)
                    [2/7] Injecting future query boundary date {forecast_date.strftime('%Y-%m-%d')}
                    [3/7] Re-calculating stationary Lags (1, 7, 14, 30 days)
                    [4/7] Re-calculating Rolling Mean & Standard Deviation (window=7, window=14)
                    [5/7] Performing O(1) metadata join (StoreType, Assortment, Distance)
                    [6/7] Computing cyclical sine/cosine trigonometric transformations
                    [7/7] Dispatching X_infer matrix payload to XGBoost Hist-estimator
                    >>> Turnover Predicted: €{result['predicted_sales']:,.2f}
                    >>> Operations Directive compiled successfully.
                    ```
                    """)
            else:
                st.error(f"❌ Forecast retrieval failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Connection error: Could not reach forecasting microservice at {api_url_input}. Error: {str(e)}")
