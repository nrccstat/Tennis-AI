import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

if not st.session_state.get("page_config_set", False):
    st.set_page_config(
        page_title="Tennis Match Predictor",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.session_state["page_config_set"] = True

if "player1" in st.session_state and "player2" in st.session_state:
    if st.session_state.player1 == st.session_state.player2:
        submitted = False

submitted = False
if "submitted" in locals():
    if submitted and st.session_state.get("player1", None) == st.session_state.get("player2", None):
        st.error("Player 1 and Player 2 cannot be the same. Please select two different players.")
        submitted = False

st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #e3f0ff 0%, #c9e7f2 100%) !important;
    }
    .block-container {
        background: rgba(220, 235, 250, 0.92) !important;
        border-radius: 18px;
        border-left: 8px solid #6bb3f2;
        border-right: 8px solid #4f8cff;
        padding: 1.5em 1em;
    }
    .stButton>button {
        color: #fff;
        background: linear-gradient(90deg, #4f8cff 0%, #6bb3f2 100%);
        border-radius: 12px;
        padding: 0.7em 2.5em;
        font-weight: bold;
        font-size: 1.15em;
        box-shadow: 0 4px 16px rgba(79,140,255,0.10);
        border: none;
        transition: 0.2s;
    }
    .stButton>button:active {
        background: linear-gradient(90deg, #4f8cff 0%, #6bb3f2 100%) !important;
        color: #fff !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #6bb3f2 0%, #4f8cff 100%);
        color: #fff;
        box-shadow: 0 6px 24px rgba(79,179,255,0.18);
    }
    .stSelectbox>div>div>div>div {
        border-radius: 12px;
        font-size: 1.08em;
        background: linear-gradient(90deg, #e3f0ff 0%, #c9e7f2 100%);
        color: #333;
    }
    .stMetric {
        background: linear-gradient(90deg, #4f8cff 0%, #6bb3f2 100%);
        border-radius: 16px;
        padding: 1.2em;
        margin-bottom: 1.2em;
        box-shadow: 0 4px 16px rgba(79,140,255,0.10);
        color: #fff !important;
    }
    .stDataFrame, .stTable {
        background: linear-gradient(90deg, #fff 0%, #e3f0ff 100%);
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(56,232,255,0.07);
    }
    .stAlert {
        border-radius: 14px;
        background: linear-gradient(90deg, #6bb3f2 0%, #4f8cff 100%);
        color: #fff !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #4f8cff, #6bb3f2, #c9e7f2, #e3f0ff);
        border-radius: 10px;
    }
    .stSubheader, .stHeader, .stTitle {
        color: #4f8cff !important;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #6bb3f2;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4f8cff !important;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #6bb3f2;
    }
    /* Card effect for form */
    .stForm {
        background: linear-gradient(120deg, #e3f0ff 60%, #c9e7f2 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(79,140,255,0.10);
        padding: 2em 2em 1em 2em;
        margin-bottom: 2em;
        border: 2px solid #6bb3f2;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #e3f0ff;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(120deg, #6bb3f2 0%, #4f8cff 100%);
        border-radius: 8px;
    }
    /* Animated gradient border for metrics */
    .stMetric {
        border: 3px solid;
        border-image: linear-gradient(90deg, #4f8cff, #6bb3f2, #c9e7f2, #e3f0ff) 1;
        animation: borderMove 3s linear infinite;
    }
    @keyframes borderMove {
        0% { border-image-source: linear-gradient(90deg, #4f8cff, #6bb3f2, #c9e7f2, #e3f0ff);}
        100% { border-image-source: linear-gradient(270deg, #4f8cff, #6bb3f2, #c9e7f2, #e3f0ff);}
    }
    .stTitle {
        text-shadow: 0 0 8px #6bb3f2, 0 0 2px #4f8cff;
    }
    .stTable th {
        background: linear-gradient(90deg, #4f8cff 0%, #6bb3f2 100%);
        color: #fff;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .stApp {
        background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .stButton>button {
        color: #fff;
        background: linear-gradient(90deg, #ff4b4b 0%, #ffb347 100%);
        border-radius: 12px;
        padding: 0.7em 2.5em;
        font-weight: bold;
        font-size: 1.15em;
        box-shadow: 0 4px 16px rgba(255,75,75,0.15);
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ffb347 0%, #ff4b4b 100%);
        color: #fff;
        box-shadow: 0 6px 24px rgba(255,179,71,0.18);
    }
    .stSelectbox>div>div>div>div {
        border-radius: 12px;
        font-size: 1.08em;
        background: linear-gradient(90deg, #e0e7ef 0%, #f8fafc 100%);
        color: #333;
    }
    .stMetric {
        background: linear-gradient(90deg, #4f8cff 0%, #38e8ff 100%);
        border-radius: 16px;
        padding: 1.2em;
        margin-bottom: 1.2em;
        box-shadow: 0 4px 16px rgba(79,140,255,0.10);
        color: #fff !important;
    }
    .stDataFrame, .stTable {
        background: linear-gradient(90deg, #fff 0%, #e0e7ef 100%);
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(56,232,255,0.07);
    }
    .stAlert {
        border-radius: 14px;
        background: linear-gradient(90deg, #ffb347 0%, #ff4b4b 100%);
        color: #fff !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #4f8cff, #38e8ff, #ffb347, #ff4b4b);
        border-radius: 10px;
    }
    .stSubheader, .stHeader, .stTitle {
        color: #4f8cff !important;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #ffb347;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ff4b4b !important;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #4f8cff;
    }
    /* Card effect for form */
    .stForm {
        background: linear-gradient(120deg, #f8fafc 60%, #e0e7ef 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(79,140,255,0.10);
        padding: 2em 2em 1em 2em;
        margin-bottom: 2em;
        border: 2px solid #ffb347;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #e0e7ef;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(120deg, #ffb347 0%, #ff4b4b 100%);
        border-radius: 8px;
    }
    /* Animated gradient border for metrics */
    .stMetric {
        border: 3px solid;
        border-image: linear-gradient(90deg, #4f8cff, #38e8ff, #ffb347, #ff4b4b) 1;
        animation: borderMove 3s linear infinite;
    }
    @keyframes borderMove {
        0% { border-image-source: linear-gradient(90deg, #4f8cff, #38e8ff, #ffb347, #ff4b4b);}
        100% { border-image-source: linear-gradient(270deg, #4f8cff, #38e8ff, #ffb347, #ff4b4b);}
    }
    /* Add a subtle glow to the title */
    .stTitle {
        text-shadow: 0 0 8px #ffb347, 0 0 2px #4f8cff;
    }
    /* Add color to table headers */
    .stTable th {
        background: linear-gradient(90deg, #4f8cff 0%, #38e8ff 100%);
        color: #fff;
        font-weight: bold;
    }
    /* Add a colored border to columns */
    .block-container {
        border-left: 8px solid #ffb347;
        border-right: 8px solid #4f8cff;
        border-radius: 18px;
        padding: 1.5em 1em;
        background: linear-gradient(120deg, #f8fafc 80%, #e0e7ef 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tennis Match Predictor")
st.markdown("""
Predict the outcome of tennis matches using our machine learning model!
Enter the names of both players and the court surface to get a prediction.
""")

@st.cache_data
def load_data():
    data_dir = 'TML-Database/'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    df_list = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values('tourney_date')
    from src.feature_engineering import calculate_elo
    df = calculate_elo(df)
    return df

@st.cache_resource
def load_model():
    model = pickle.load(open('models/tennis_model.pkl', 'rb'))
    encoder = pickle.load(open('models/encoder.pkl', 'rb'))
    feature_columns = pickle.load(open('models/feature_columns.pkl', 'rb'))
    base_features = [col for col in feature_columns if not col.startswith('surface_')]
    return model, encoder, base_features

df = load_data()
all_players = sorted(set(df['winner_name'].unique()) | set(df['loser_name'].unique()))

def get_player_stats(player_name, df):
    player_matches = df[(df['winner_name'] == player_name) | (df['loser_name'] == player_name)]
    if len(player_matches) == 0:
        return None
    recent_match = player_matches.iloc[-1]
    is_winner = recent_match['winner_name'] == player_name
    return {
        'elo': recent_match['winner_elo'] if is_winner else recent_match['loser_elo'],
        'age': recent_match['winner_age'] if is_winner else recent_match['loser_age'],
        'height': recent_match['winner_ht'] if is_winner else recent_match['loser_ht']
    }

with st.form("match_prediction_form"):
    st.subheader("Match Details")
    court_type = st.selectbox(
        "Court Surface",
        ["Hard", "Clay", "Grass", "Carpet"]
    )
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", all_players)
    with col2:
        player2 = st.selectbox("Player 2", all_players)
    submitted = st.form_submit_button("Predict Match Outcome")

if submitted:
    if player1 == player2:
        st.error("Player 1 and Player 2 cannot be the same. Please select two different players.")
    else:
        try:
            model, encoder, base_features = load_model()
            p1_stats = get_player_stats(player1, df)
            p2_stats = get_player_stats(player2, df)
            if p1_stats is None or p2_stats is None:
                st.error("Could not find historical data for one or both players.")
            else:
                input_data = pd.DataFrame({
                    'surface': [court_type],
                    'winner_elo': [p1_stats['elo']],
                    'loser_elo': [p2_stats['elo']],
                    'winner_age': [p1_stats['age']],
                    'winner_ht': [p1_stats['height']],
                    'loser_age': [p2_stats['age']],
                    'loser_ht': [p2_stats['height']],
                    'winner_recent_form': [p1_stats['elo']],  
                    'loser_recent_form': [p2_stats['elo']],   
                    'winner_surface_win_rate': [0.5],  
                    'loser_surface_win_rate': [0.5],   
                    'h2h_matches': [0],               
                    'winner_tourney_wins': [0],        
                    'loser_tourney_wins': [0],         
                    'winner_days_since_last_match': [0], 
                    'loser_days_since_last_match': [0]   
                })
                
                numeric_columns = [
                    'winner_elo', 'loser_elo', 'winner_age', 'winner_ht', 'loser_age', 'loser_ht',
                    'winner_recent_form', 'loser_recent_form', 'winner_surface_win_rate', 'loser_surface_win_rate',
                    'h2h_matches', 'winner_tourney_wins', 'loser_tourney_wins',
                    'winner_days_since_last_match', 'loser_days_since_last_match'
                ]
                
                cat_features = ['surface']
                X_cat = input_data[cat_features]
                X_cat_encoded = encoder.transform(X_cat)
                
                input_data[numeric_columns] = input_data[numeric_columns].fillna(0)
                X_num = input_data[numeric_columns].values
                X = np.hstack([X_num, X_cat_encoded])
                prob = model.predict_proba(X)[0]
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=f"{player1} Win Probability",
                        value=f"{prob[1]*100:.1f}%"
                    )
                with col2:
                    st.metric(
                        label=f"{player2} Win Probability",
                        value=f"{prob[0]*100:.1f}%"
                    )
                st.progress(prob[1])
                st.subheader("Match Insights")
                if prob[1] > 0.7:
                    st.success(f"Strong favorite: {player1}")
                elif prob[1] < 0.3:
                    st.success(f"Strong favorite: {player2}")
                else:
                    st.info("This is a close match!")
                st.subheader("Player Statistics")
                stats_df = pd.DataFrame({
                    'Player': [player1, player2],
                    'Elo': [p1_stats['elo'], p2_stats['elo']],
                    'Age': [p1_stats['age'], p2_stats['age']],
                    'Height (cm)': [p1_stats['height'], p2_stats['height']]
                })
                st.table(stats_df)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure the model files are available in the 'models' directory.")

st.markdown("---")
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .stApp {
        background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .stButton>button {
        color: #fff;
        background: linear-gradient(90deg, #ff4b4b 0%, #ffb347 100%);
        border-radius: 12px;
        padding: 0.7em 2.5em;
        font-weight: bold;
        font-size: 1.15em;
        box-shadow: 0 4px 16px rgba(255,75,75,0.15);
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ffb347 0%, #ff4b4b 100%);
        color: #fff;
        box-shadow: 0 6px 24px rgba(255,179,71,0.18);
    }
    .stSelectbox>div>div>div>div {
        border-radius: 12px;
        font-size: 1.08em;
        background: linear-gradient(90deg, #e0e7ef 0%, #f8fafc 100%);
        color: #333;
    }
    .stMetric {
        background: linear-gradient(90deg, #4f8cff 0%, #38e8ff 100%);
        border-radius: 16px;
        padding: 1.2em;
        margin-bottom: 1.2em;
        box-shadow: 0 4px 16px rgba(79,140,255,0.10);
        color: #fff !important;
    }
    .stDataFrame, .stTable {
        background: linear-gradient(90deg, #fff 0%, #e0e7ef 100%);
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(56,232,255,0.07);
    }
    .stAlert {
        border-radius: 14px;
        background: linear-gradient(90deg, #ffb347 0%, #ff4b4b 100%);
        color: #fff !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #4f8cff, #38e8ff, #ffb347, #ff4b4b);
        border-radius: 10px;
    }
    .stSubheader, .stHeader, .stTitle {
        color: #4f8cff !important;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #ffb347;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ff4b4b !important;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #4f8cff;
# The error for same player selection is now handled above, so this is no longer needed.
""", unsafe_allow_html=True)
st.markdown("© 2024 <span style='color:#ff4b4b;font-weight:bold;'>Tennis Match Predictor</span>. All rights reserved.", unsafe_allow_html=True)