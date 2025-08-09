import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import math
import random

# --- Initial Page Configuration ---
st.set_page_config(
    page_title="Tennis Pro-Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* General Body and App Styling */
        .stApp {
            background: #f0f2f6;
        }

        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Custom Button Styling */
        .stButton>button {
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: bold;
            font-size: 1.1rem;
            color: white;
            background-color: #0072b5;
            border: none;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #005f99;
            transform: scale(1.02);
            box-shadow: 0 6px 16px rgba(0,114,181,0.2);
        }

        /* Title and Header Styling */
        .stTitle, h1, h2, h3, h4 {
            color: #1E293B;
        }
        h1 {
            text-align: center;
        }

        /* Metric Styling */
        .stMetric {
            background-color: #F8F9FA;
            border-left: 5px solid #0072b5;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            justify-content: center; /* Center the tabs */
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 8px;
            padding: 10px 20px;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0072b5;
            color: white;
            font-weight: bold;
        }

        /* --- BRACKET STYLING --- */
        /* This is the new container that allows wrapping */
        .bracket-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 16px; /* Space between matchups */
        }

        .bracket-matchup {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            flex: 1 1 200px; /* Flex properties: grow, shrink, basis */
            max-width: 250px; /* Prevents items from becoming too wide */
        }
        .bracket-winner {
            font-weight: bold;
            color: #28a745;
        }
        .bracket-prob {
            font-size: 0.85em;
            color: #6c757d;
        }

        /* Style for Selectbox background */
        .stSelectbox>div>div>div {
            background-color: white !important;
            color: #333 !important;
            border-radius: 8px;
            border: 1px solid #ced4da;
        }
    </style>
""", unsafe_allow_html=True)


# --- Data and Model Loading (Cached) ---
@st.cache_data
def load_data():
    """Loads and preprocesses tennis data, calculating Elo ratings."""
    try:
        data_dir = 'TML-Database/'
        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        if not csv_files:
            st.error("No CSV files found in the 'TML-Database' directory. Please add your data files.")
            return pd.DataFrame(), []

        df_list = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
        df = pd.concat(df_list, ignore_index=True)
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        df = df.sort_values('tourney_date')

        from src.feature_engineering import calculate_elo
        df = calculate_elo(df)

        all_players = sorted(pd.concat([df['winner_name'], df['loser_name']]).dropna().unique())
        return df, all_players
    except (FileNotFoundError, ImportError) as e:
        st.error(f"Error loading data or dependencies: {e}. Please ensure `TML-Database` and `src/feature_engineering.py` are correctly placed.")
        return pd.DataFrame(), []

@st.cache_resource
def load_model():
    """Loads the pickled model, encoder, and feature columns."""
    try:
        with open('models/tennis_model.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
        with open('models/encoder.pkl', 'rb') as f_encoder:
            encoder = pickle.load(f_encoder)
        with open('models/feature_columns.pkl', 'rb') as f_cols:
            feature_columns = pickle.load(f_cols)
        return model, encoder, feature_columns
    except FileNotFoundError:
        st.error("Model files not found in the 'models' directory.")
        return None, None, None

# --- NEW: A single function to initialize the app, which we can cache ---
@st.cache_resource (show_spinner=False)
def initialize_app():
    """Load all data and models. This function will be cached."""
    df, all_players = load_data()
    model, encoder, feature_columns = load_model()
    return df, all_players, model, encoder, feature_columns

# --- Use st.spinner for the initial load ---
with st.spinner("Loading..."):
    df, all_players, model, encoder, feature_columns = initialize_app()

# If loading fails, stop the app
if df is None or all_players is None or model is None:
    st.error("Application failed to initialize. Please check the data and model files.")
    st.stop()


# --- Core Logic Functions ---
def get_player_stats(player_name, df):
    """Retrieves the latest stats for a given player."""
    player_matches = df[(df['winner_name'] == player_name) | (df['loser_name'] == player_name)]
    if player_matches.empty: return None
    last_match = player_matches.iloc[-1]
    is_winner = last_match['winner_name'] == player_name
    return {
        'elo': last_match['winner_elo'] if is_winner else last_match['loser_elo'],
        'age': last_match['winner_age'] if is_winner else last_match['loser_age'],
        'height': last_match['winner_ht'] if is_winner else last_match['loser_ht']
    }

def predict_match(player1_name, player2_name, court_type, df, model, encoder, feature_columns):
    """Predicts the outcome of a match between two players."""
    p1_stats = get_player_stats(player1_name, df)
    p2_stats = get_player_stats(player2_name, df)

    if not p1_stats or not p2_stats:
        return None, None, "Could not find historical data for one or both players."

    input_data = pd.DataFrame({
        'surface': [court_type], 'winner_elo': [p1_stats['elo']], 'loser_elo': [p2_stats['elo']],
        'winner_age': [p1_stats['age']], 'loser_age': [p2_stats['age']],
        'winner_ht': [p1_stats['height']], 'loser_ht': [p2_stats['height']],
        'winner_recent_form': [p1_stats['elo']], 'loser_recent_form': [p2_stats['elo']],
        'winner_surface_win_rate': [0.5], 'loser_surface_win_rate': [0.5], 'h2h_matches': [0],
        'winner_tourney_wins': [0], 'loser_tourney_wins': [0],
        'winner_days_since_last_match': [0], 'loser_days_since_last_match': [0]
    })

    numeric_cols = [col for col in feature_columns if not col.startswith('surface_')]
    input_numeric = input_data.reindex(columns=numeric_cols).fillna(0)
    X_cat_encoded = encoder.transform(input_data[['surface']])
    X = np.hstack([input_numeric.values, X_cat_encoded])
    probability = model.predict_proba(X)[0]
    return probability, (p1_stats, p2_stats), None

# --- Main Application ---
st.title("🎾 Tennis Pro-Predictor")
st.markdown("<h3 style='text-align: center; color: #4A5568;'>From Head-to-Head Showdowns to Full Tournament Simulations</h3>", unsafe_allow_html=True)

# The data is already loaded above, so we can now display the UI
if df.empty or not model:
    st.warning("Application cannot start due to missing data or model files.")
    st.stop()

# --- UI Tabs ---
tab1, tab2 = st.tabs(["⚡ Head-to-Head", "🏆 Tournament Bracket"])

# ============================
#  Tab 1: Head-to-Head
# ============================
with tab1:
    st.subheader("Match Setup")
    p1 = st.selectbox("Select Player 1", all_players, index=12, key="h2h_p1")
    p2 = st.selectbox("Select Player 2", all_players, index=34, key="h2h_p2")
    court = st.selectbox("Select Court Surface", ["Hard", "Clay", "Grass", "Carpet"], key="h2h_court")

    _, btn_col, _ = st.columns([2.5, 1, 2.5])
    with btn_col:
        predict_button = st.button("Predict Outcome", key="h2h_predict", use_container_width=True)

    if predict_button:
        st.markdown("---")
        if p1 == p2:
            st.error("Players must be different. Please select two unique players.")
        else:
            prob, stats, error_msg = predict_match(p1, p2, court, df, model, encoder, feature_columns)
            if error_msg:
                st.error(error_msg)
            else:
                p1_stats, p2_stats = stats
                p1_prob, p2_prob = prob[1], prob[0]
                st.subheader("📈 Prediction Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label=f"{p1} Win Probability", value=f"{p1_prob*100:.1f}%")
                    st.progress(p1_prob)
                with res_col2:
                    st.metric(label=f"{p2} Win Probability", value=f"{p2_prob*100:.1f}%")
                    st.progress(p2_prob)

                st.subheader("📊 Head-to-Head Stats")
                stats_df = pd.DataFrame([
                    {'Player': p1, 'Elo Rating': f"{p1_stats['elo']:.0f}", 'Age': f"{p1_stats['age']:.1f}", 'Height (cm)': f"{p1_stats['height']:.0f}"},
                    {'Player': p2, 'Elo Rating': f"{p2_stats['elo']:.0f}", 'Age': f"{p2_stats['age']:.1f}", 'Height (cm)': f"{p2_stats['height']:.0f}"}
                ]).set_index('Player')
                st.dataframe(stats_df, use_container_width=True)

# ============================
#  Tab 2: Tournament Bracket
# ============================
with tab2:
    st.subheader("Build Your Bracket")
    tourney_court = st.selectbox("Select Tournament Court Surface", ["Hard", "Clay", "Grass", "Carpet"], key="tourney_court")
    num_players = st.radio("Select Bracket Size", [4, 8, 16, 32], index=1, horizontal=True, key="tourney_size")

    st.info(f"Select **{num_players}** unique players to enter the tournament.")

    if 'tourney_players' not in st.session_state or len(st.session_state.tourney_players) != num_players:
        st.session_state.tourney_players = [all_players[i * 10 % len(all_players)] for i in range(num_players)]

    cols = st.columns(4)
    for i in range(num_players):
        st.session_state.tourney_players[i] = cols[i % 4].selectbox(
            f"Player {i+1}", all_players,
            index=all_players.index(st.session_state.tourney_players[i]),
            key=f"tourney_p{i}"
        )

    _, btn_col_tourney, _ = st.columns([2.5, 1, 2.5])
    with btn_col_tourney:
        simulate_button = st.button("Simulate Tournament", key="tourney_sim", use_container_width=True)

    if simulate_button:
        st.markdown("---")
        selected_players = st.session_state.tourney_players
        if len(set(selected_players)) != num_players:
            st.error("All selected players must be unique. Please check your selections.")
        else:
            random.shuffle(selected_players)
            st.subheader("🏆 Tournament Results")

            round_names = {32: "Round of 32", 16: "Round of 16", 8: "Quarterfinals", 4: "Semifinals", 2: "Final"}
            current_round_players = selected_players
            current_round_size = len(current_round_players)

            while current_round_size >= 2:
                round_name = round_names.get(current_round_size, f"Round of {current_round_size}")
                st.markdown(f"#### {round_name}")

                next_round_winners = []
                matchups = [current_round_players[i:i+2] for i in range(0, len(current_round_players), 2)]
                
                # Use the new flexible container instead of st.columns
                matchup_html = ""
                for matchup in matchups:
                    p1, p2 = matchup[0], matchup[1]
                    prob, _, error_msg = predict_match(p1, p2, tourney_court, df, model, encoder, feature_columns)

                    if error_msg:
                        st.warning(f"Sim Error: {p1} vs {p2}")
                        winner, p1_prob, p2_prob = random.choice([p1, p2]), 0.5, 0.5
                    else:
                        p1_prob, p2_prob = prob[1], prob[0]
                        winner = p1 if p1_prob > p2_prob else p2

                    next_round_winners.append(winner)
                    
                    matchup_html += (f'<div class="bracket-matchup">'
                                     f'<div>{p1}<span class="bracket-prob"> ({p1_prob*100:.1f}%)</span></div>'
                                     f'<div style="margin: 2px 0; font-size: 0.8em;">vs</div>'
                                     f'<div>{p2}<span class="bracket-prob"> ({p2_prob*100:.1f}%)</span></div>'
                                     f'<hr style="margin: 6px 0;">'
                                     f'<div class="bracket-winner">🏆 {winner}</div>'
                                     f'</div>')
                
                # Display all matchups inside the wrapping container
                st.markdown(f'<div class="bracket-container">{matchup_html}</div>', unsafe_allow_html=True)

                current_round_players = next_round_winners
                current_round_size = len(current_round_players)
                if current_round_size >= 2: st.markdown("---")

            st.markdown(f"<h2 style='text-align:center; color: #0072b5;'>🎉 Tournament Champion 🎉</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center;'>{current_round_players[0]}</h1>", unsafe_allow_html=True)
            st.balloons()


# --- Footer ---
st.markdown("---")
st.markdown("© 2025 <span style='color:#0072b5;font-weight:bold;'>Tennis Pro-Predictor</span>. Model based on historical ATP data.", unsafe_allow_html=True)