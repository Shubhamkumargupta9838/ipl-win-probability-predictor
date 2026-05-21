import pickle
from base64 import b64encode
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPL Victory Predictor", layout="wide")

pipe = pickle.load(open("pipe.pkl", "rb"))
encoder = pipe.named_steps["step1"].named_transformers_["trf"]
teams = sorted(set(encoder.categories_[0]).union(set(encoder.categories_[1])))
cities = sorted(encoder.categories_[2])


def parse_int_input(label, value, min_value=None, max_value=None):
    try:
        parsed_value = int(value)
    except ValueError:
        st.error(f"{label} must be a whole number.")
        st.stop()

    if min_value is not None and parsed_value < min_value:
        st.error(f"{label} must be at least {min_value}.")
        st.stop()

    if max_value is not None and parsed_value > max_value:
        st.error(f"{label} must be at most {max_value}.")
        st.stop()

    return parsed_value


def parse_float_input(label, value, min_value=None, max_value=None):
    try:
        parsed_value = float(value)
    except ValueError:
        st.error(f"{label} must be a number.")
        st.stop()

    if min_value is not None and parsed_value < min_value:
        st.error(f"{label} must be at least {min_value}.")
        st.stop()

    if max_value is not None and parsed_value > max_value:
        st.error(f"{label} must be at most {max_value}.")
        st.stop()

    return parsed_value

BACKGROUND_IMAGE = (
    "https://images.unsplash.com/photo-1540747913346-19e32dc3e97e"
    "?auto=format&fit=crop&w=1600&q=80"
)
IPL_LOGO_PATH = Path("ipl-logo.png")
IPL_LOGO_DATA = b64encode(IPL_LOGO_PATH.read_bytes()).decode("utf-8")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&display=swap');

    .stApp {{
        background:
            linear-gradient(180deg, rgba(3, 7, 24, 0.70), rgba(3, 7, 24, 0.74)),
            url("{BACKGROUND_IMAGE}") center center / cover no-repeat fixed;
        font-family: 'Barlow', sans-serif;
    }}

    .main .block-container {{
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stToolbar"] {{
        right: 1rem;
    }}

    .hero-card {{
        padding: 1.2rem 1.4rem 0.8rem 1.4rem;
        border-radius: 28px;
        background: linear-gradient(180deg, rgba(9, 18, 44, 0.78), rgba(12, 22, 36, 0.56));
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(10px);
    }}

    .hero-content {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
    }}

    .hero-copy {{
        flex: 1 1 auto;
        min-width: 0;
    }}

    .hero-logo {{
        width: min(260px, 30vw);
        min-width: 150px;
        padding: 0.4rem 0.2rem;
        object-fit: contain;
        filter: drop-shadow(0 20px 34px rgba(37, 99, 235, 0.28));
    }}

    .hero-title {{
        color: #ffffff;
        font-size: 4rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin: 0 0 0.45rem 0;
        text-transform: uppercase;
    }}

    .hero-subtitle {{
        color: rgba(233, 239, 255, 0.9);
        font-size: 1.05rem;
        margin-bottom: 1.35rem;
    }}

    .panel-title {{
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 0.4rem 0 1rem 0;
    }}

    .stMarkdown p {{
        font-family: 'Barlow', sans-serif;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {{
        min-height: 58px;
        border-radius: 16px;
        background: rgba(31, 41, 55, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.10);
        color: #ffffff;
        box-shadow: none;
    }}

    div[data-baseweb="select"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {{
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }}

    label[data-testid="stWidgetLabel"] p {{
        color: #e5e7eb;
        font-size: 0.96rem;
        font-weight: 600;
    }}

    .stButton > button {{
        width: 100%;
        min-height: 60px;
        border: 0;
        border-radius: 18px;
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: #ffffff;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 18px 30px rgba(22, 163, 74, 0.25);
    }}

    .stButton > button:hover {{
        background: linear-gradient(90deg, #15803d, #16a34a);
        color: #ffffff;
    }}

    .results-wrap {{
        margin-top: 1.25rem;
        padding: 1rem;
        border-radius: 22px;
        background: rgba(7, 15, 28, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(8px);
    }}

    .results-title {{
        color: #dbeafe;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.9rem;
    }}

    .result-card {{
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }}

    .result-team {{
        color: #d1d5db;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }}

    .result-value {{
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
    }}

    @media (max-width: 900px) {{
        .hero-content {{
            align-items: flex-start;
            flex-direction: column-reverse;
            gap: 1rem;
        }}

        .hero-logo {{
            width: min(220px, 70vw);
            min-width: 0;
        }}

        .hero-title {{
            font-size: 2.7rem;
        }}

        .main .block-container {{
            padding-top: 1.2rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-copy">
                <div class="hero-title">IPL Victory Predictor</div>
                <div class="hero-subtitle">
                    Pick the batting side, bowling side, venue, and live chase situation to estimate the winning chance.
                </div>
            </div>
            <img class="hero-logo" src="data:image/png;base64,{IPL_LOGO_DATA}" alt="IPL logo">
        </div>
        <div class="panel-title">Match Situation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select Batting Team", teams, index=None, placeholder=" select the batting team ")
with col2:
    bowling_team = st.selectbox("Select Bowling Team", teams, index=None, placeholder=" select the batting team ")

selected_city = st.selectbox("Select Venue", cities)
target_input = st.text_input("Target", value="0")

col3, col4, col5 = st.columns(3)
with col3:
    score_input = st.text_input("Current Score", value="0")
with col4:
    overs_input = st.text_input("Over Completed", value="0.0")
with col5:
    wickets_down_input = st.text_input("Total Wickets Down", value="0")

if st.button("Predict Victory"):
    target = parse_int_input("Target", target_input, min_value=0)
    score = parse_int_input("Score", score_input, min_value=0)
    overs = parse_float_input("Overs completed", overs_input, min_value=0.0, max_value=19.5)
    wickets_down = parse_int_input("Wickets down", wickets_down_input, min_value=0, max_value=9)

    if batting_team is None or bowling_team is None:
        st.error("Select both batting and bowling teams.")
        st.stop()

    if batting_team == bowling_team:
        st.error("Batting team and bowling team must be different.")
        st.stop()

    overs_whole = int(overs)
    overs_ball = round((overs - overs_whole) * 10)
    legal_balls = overs_whole * 6 + overs_ball

    if overs_ball > 5:
        st.error("Use cricket overs format like 10.2 or 17.5. The digit after the decimal cannot be more than 5.")
        st.stop()

    if legal_balls <= 0:
        st.error("Overs completed must be greater than 0.")
        st.stop()

    if legal_balls >= 120:
        st.error("Overs completed must be less than 20 overs.")
        st.stop()

    runs_left = target - score
    balls_left = 120 - legal_balls
    wickets_in_hand = 10 - wickets_down
    crr = score * 6 / legal_balls
    rrr = runs_left * 6 / balls_left if balls_left else 0

    match_df = pd.DataFrame(
        {
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [selected_city],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets": [wickets_in_hand],
            "total_runs_x": [target],
            "crr": [crr],
            "rrr": [rrr],
        }
    )

    result = pipe.predict_proba(match_df)
    bowling_win = round(result[0][0] * 100)
    batting_win = round(result[0][1] * 100)

    st.markdown('<div class="results-wrap"><div class="results-title">Win Probability</div>', unsafe_allow_html=True)
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-team">{batting_team}</div>
                <div class="result-value">{batting_win}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-team">{bowling_team}</div>
                <div class="result-value">{bowling_win}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
