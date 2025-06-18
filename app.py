import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

st.set_page_config(
    page_title="Tennis Match Predictor",
    page_icon="🎾",
    layout="wide"
)

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
    # Filter out the one-hot encoded surface columns
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
    try:
        model, encoder, base_features = load_model()
        p1_stats = get_player_stats(player1, df)
        p2_stats = get_player_stats(player2, df)
        if p1_stats is None or p2_stats is None:
            st.error("Could not find historical data for one or both players.")
        else:
            # Create input data with all features
            input_data = pd.DataFrame({
                'surface': [court_type],
                'winner_elo': [p1_stats['elo']],
                'loser_elo': [p2_stats['elo']],
                'winner_age': [p1_stats['age']],
                'winner_ht': [p1_stats['height']],
                'loser_age': [p2_stats['age']],
                'loser_ht': [p2_stats['height']],
                'winner_recent_form': [p1_stats['elo']],  # Using current ELO as recent form
                'loser_recent_form': [p2_stats['elo']],   # Using current ELO as recent form
                'winner_surface_win_rate': [0.5],  # Default to 0.5 if not available
                'loser_surface_win_rate': [0.5],   # Default to 0.5 if not available
                'h2h_matches': [0],                # Default to 0 if not available
                'winner_tourney_wins': [0],        # Default to 0 if not available
                'loser_tourney_wins': [0],         # Default to 0 if not available
                'winner_days_since_last_match': [0], # Default to 0 if not available
                'loser_days_since_last_match': [0]   # Default to 0 if not available
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
st.markdown("Built with using Streamlit and Machine Learning") 