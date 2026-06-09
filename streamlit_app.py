# ============================================================
# FILE: app/streamlit_app.py
# PURPOSE: Interactive web app for scheme recommendations
# RUN: streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title = "🏛️ Govt Scheme Recommender",
    page_icon  = "🇮🇳",
    layout     = "wide",         # Use full browser width
    initial_sidebar_state = "expanded"
)

# ── Custom CSS Styling ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: bold;
        color: #FF6B00; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center; color: #555; font-size: 1rem; margin-bottom: 2rem;
    }
    .scheme-card {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 5px solid #2e7d32;
        padding: 15px; border-radius: 10px; margin: 8px 0;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .no-scheme {
        background: #fff3e0; border-left: 5px solid #e65100;
        padding: 15px; border-radius: 10px;
    }
    .metric-box {
        background: #e3f2fd; border-radius: 8px;
        padding: 10px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)
# unsafe_allow_html=True: allows raw HTML/CSS injection

# ── Load Model and Preprocessing Tools ───────────────────────
@st.cache_resource
# @st.cache_resource: loads model only ONCE, not on every user click
# This makes the app much faster
def load_artifacts():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to locate the repository root that contains the 'models' folder.
    # If this file is in an 'app/' subfolder, the parent directory will be the repo root.
    if os.path.exists(os.path.join(file_dir, 'models')):
        base = file_dir
    elif os.path.exists(os.path.join(os.path.dirname(file_dir), 'models')):
        base = os.path.dirname(file_dir)
    else:
        # Fallback: use the file's directory (keeps behavior stable when running from repo root)
        base = file_dir

    model_path    = os.path.join(base, 'models', 'tuned_random_forest.pkl')
    encoder_path  = os.path.join(base, 'models', 'encoders.pkl')
    scaler_path   = os.path.join(base, 'models', 'scaler.pkl')
    labels_path   = os.path.join(base, 'models', 'label_cols.pkl')

    # Fallback to untuned model if tuned doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(base, 'models', 'best_model.pkl')

    with open(model_path,   'rb') as f: model    = pickle.load(f)
    with open(encoder_path, 'rb') as f: encoders = pickle.load(f)
    with open(scaler_path,  'rb') as f: scaler   = pickle.load(f)
    with open(labels_path,  'rb') as f: labels   = pickle.load(f)
    # 'rb' = read binary mode

    return model, encoders, scaler, labels

model, encoders, scaler, LABEL_COLS = load_artifacts()

# ── Scheme Details Dictionary ──────────────────────────────────
SCHEME_INFO = {
    'PM_KISAN': {
        'full_name'  : 'PM Kisan Samman Nidhi',
        'benefit'    : '₹6,000/year direct transfer to farmer bank accounts',
        'ministry'   : 'Ministry of Agriculture',
        'apply_link' : 'https://pmkisan.gov.in',
        'icon'       : '🌾'
    },
    'PMAY': {
        'full_name'  : 'PM Awas Yojana (Housing for All)',
        'benefit'    : 'Subsidy up to ₹2.67 Lakh for housing construction',
        'ministry'   : 'Ministry of Housing & Urban Affairs',
        'apply_link' : 'https://pmaymis.gov.in',
        'icon'       : '🏠'
    },
    'Ayushman_Bharat': {
        'full_name'  : 'Ayushman Bharat PM-JAY',
        'benefit'    : 'Health coverage up to ₹5 Lakh per family per year',
        'ministry'   : 'Ministry of Health & Family Welfare',
        'apply_link' : 'https://pmjay.gov.in',
        'icon'       : '🏥'
    },
    'PM_Scholarship': {
        'full_name'  : 'PM Scholarship Scheme',
        'benefit'    : '₹2,500–₹3,000/month for higher education',
        'ministry'   : 'Ministry of Education',
        'apply_link' : 'https://scholarships.gov.in',
        'icon'       : '🎓'
    },
    'Kisan_Credit_Card': {
        'full_name'  : 'Kisan Credit Card Scheme',
        'benefit'    : 'Short-term crop loan at 4% interest rate',
        'ministry'   : 'Ministry of Agriculture',
        'apply_link' : 'https://pmkisan.gov.in/kcc',
        'icon'       : '💳'
    },
    'PMEGP': {
        'full_name'  : 'PM Employment Generation Programme',
        'benefit'    : 'Subsidy 15–35% on project cost up to ₹25 Lakh',
        'ministry'   : 'Ministry of MSME',
        'apply_link' : 'https://kviconline.gov.in/pmegpeportal',
        'icon'       : '🏭'
    },
    'Disability_Pension': {
        'full_name'  : 'Indira Gandhi National Disability Pension',
        'benefit'    : '₹300–₹500/month pension for disabled persons',
        'ministry'   : 'Ministry of Social Justice',
        'apply_link' : 'https://nsap.nic.in',
        'icon'       : '♿'
    },
    'Minority_Scholarship': {
        'full_name'  : 'Post-Matric Scholarship (OBC/SC/ST)',
        'benefit'    : 'Full tuition fee + maintenance allowance',
        'ministry'   : 'Ministry of Social Justice / Tribal Affairs',
        'apply_link' : 'https://scholarships.gov.in',
        'icon'       : '📚'
    }
}

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏛️ AI-Based Government Scheme Recommender</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter your details below to discover schemes you qualify for</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
with st.form("user_form"):
    st.subheader("📋 Enter Your Details")

    col1, col2, col3 = st.columns(3)
    # Creates 3 side-by-side columns for a cleaner layout

    with col1:
        age = st.slider("Age", 18, 80, 30)
        # slider(label, min, max, default)

        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        # selectbox: dropdown menu

        income = st.number_input("Annual Income (₹)", 0, 1000000,
                                  step=10000, value=100000)
        # number_input(label, min, max, step, default)

        family_size = st.slider("Family Size", 1, 10, 4)

    with col2:
        occupation = st.selectbox("Occupation",
                                   ["Farmer","Student","Salaried",
                                    "Business","Unemployed"])
        education  = st.selectbox("Education Level",
                                   ["Illiterate","Primary","Secondary",
                                    "Graduate","Postgraduate"])
        category   = st.selectbox("Social Category",
                                   ["General","OBC","SC","ST"])
        state      = st.selectbox("State", [
            'Maharashtra','Uttar Pradesh','Bihar','Rajasthan',
            'Madhya Pradesh','Gujarat','West Bengal','Karnataka',
            'Tamil Nadu','Andhra Pradesh','Punjab','Haryana'
        ])

    with col3:
        is_farmer      = st.checkbox("Are you a Farmer?")
        # checkbox: True/False toggle

        has_disability = st.checkbox("Do you have a disability?")
        has_bpl_card   = st.checkbox("Do you have a BPL Card?")
        land_holding   = st.number_input("Land Holding (acres)",
                                          0.0, 20.0, 0.0, step=0.5)

    submitted = st.form_submit_button("🔍 Find My Schemes",
                                       use_container_width=True)
    # form_submit_button: triggers the block below only when clicked

# ── Process and Predict ───────────────────────────────────────
if submitted:

    # Build input dictionary matching training feature order
    input_dict = {
        'age'               : age,
        'gender'            : gender,
        'income_annual'     : income,
        'occupation'        : occupation,
        'education'         : education,
        'category'          : category,
        'state'             : state,
        'is_farmer'         : int(is_farmer),       # True→1, False→0
        'has_disability'    : int(has_disability),
        'land_holding_acres': land_holding,
        'has_bpl_card'      : int(has_bpl_card),
        'family_size'       : family_size
    }

    input_df = pd.DataFrame([input_dict])
    # Wrap in DataFrame for encoder/scaler compatibility (they expect DataFrames)

    # ── Apply same encoding as training ───────────────────────
    CAT_COLS = ['gender','occupation','education','category','state']
    NUM_COLS = ['age','income_annual','land_holding_acres','family_size']

    for col in CAT_COLS:
        try:
            input_df[col] = encoders[col].transform(input_df[col])
            # transform (not fit_transform) — use EXISTING mappings
        except ValueError:
            # If new unseen label, assign -1 (unknown)
            input_df[col] = -1

    input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])
    # Apply same scaling learned during training

    # ── Predict ────────────────────────────────────────────────
    prediction = model.predict(input_df)[0]
    # model.predict returns 2D array; [0] gets first (only) row
    # prediction = array of 0s and 1s, one per scheme

    # ── Display Results ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📢 Your Scheme Recommendations")

    eligible_schemes = [LABEL_COLS[i]
                        for i, val in enumerate(prediction) if val == 1]
    # List comprehension: collect scheme names where prediction = 1

    if eligible_schemes:
        st.success(f"🎉 You qualify for **{len(eligible_schemes)}** government scheme(s)!")

        for scheme_key in eligible_schemes:
            info = SCHEME_INFO.get(scheme_key, {})
            icon = info.get('icon', '📋')
            st.markdown(f"""
            <div class="scheme-card">
                <h4>{icon} {info.get('full_name', scheme_key)}</h4>
                <p><b>Benefit:</b> {info.get('benefit','N/A')}</p>
                <p><b>Ministry:</b> {info.get('ministry','N/A')}</p>
                <p><a href="{info.get('apply_link','#')}" target="_blank">
                   🔗 Apply Online</a></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-scheme">
            <h4>⚠️ No schemes found for current criteria.</h4>
            <p>Try adjusting your income, BPL card, or farmer status details.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Show Input Summary ─────────────────────────────────────
    with st.expander("📄 View Your Input Summary"):
        summary_cols = st.columns(4)
        # Display key inputs in a grid
        items = list(input_dict.items())
        for i, (k, v) in enumerate(items):
            with summary_cols[i % 4]:
                st.metric(label=k.replace('_',' ').title(), value=str(v))
                # .title() capitalizes each word; metric shows a highlighted value box