import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def calculate_elo(df, k=32, base_elo=1500):
    elos = {}
    elo_history = []
    for idx, row in df.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        w_elo = elos.get(w, base_elo)
        l_elo = elos.get(l, base_elo)
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        expected_l = 1 - expected_w
        w_elo_new = w_elo + k * (1 - expected_w)
        l_elo_new = l_elo + k * (0 - expected_l)
        elos[w] = w_elo_new
        elos[l] = l_elo_new
        elo_history.append({'winner_elo': w_elo, 'loser_elo': l_elo})
    elo_df = pd.DataFrame(elo_history)
    df = df.reset_index(drop=True)
    df['winner_elo'] = elo_df['winner_elo']
    df['loser_elo'] = elo_df['loser_elo']
    return df

def create_advanced_features(df):
    df['winner_recent_form'] = df.groupby('winner_name')['winner_elo'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['loser_recent_form'] = df.groupby('loser_name')['loser_elo'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['winner_grass_win_rate'] = df.groupby(['winner_name', 'surface'])['winner_elo'].transform(lambda x: (x > x.shift(1)).mean())
    df['h2h_matches'] = df.groupby(['winner_name', 'loser_name']).cumcount()
    df['winner_tourney_wins'] = df.groupby(['winner_name', 'tourney_name']).cumcount()
    return df

def train_ensemble_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble

def optimize_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=3,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def monte_carlo_simulation(model, X, n_simulations=1000):
    results = []
    for _ in range(n_simulations):
        preds = model.predict(X)
        results.append(preds)
    results = np.array(results)
    win_probs = results.mean(axis=0)
    return win_probs



