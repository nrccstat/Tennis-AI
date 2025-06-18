import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from typing import Dict, List, Tuple
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class XGBoostWrapper(xgb.XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validation_data = None
        
    def set_validation_data(self, validation_data):
        self._validation_data = validation_data
        
    def fit(self, X, y, **kwargs):
        if self._validation_data is not None:
            kwargs['eval_set'] = self._validation_data
        return super().fit(X, y, **kwargs)

def calculate_elo_parallel(chunk):
    elos = {}
    elo_history = []
    base_elo = 1500
    k = 32
    
    for _, row in chunk.iterrows():
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
    
    return pd.DataFrame(elo_history)

def calculate_elo(df: pd.DataFrame, k: int = 32, base_elo: int = 1500) -> pd.DataFrame:
    print("Calculating Elo ratings...")
    n_cores = multiprocessing.cpu_count()
    chunk_size = len(df) // n_cores
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(tqdm(executor.map(calculate_elo_parallel, chunks), total=len(chunks), desc="Processing Elo chunks"))
    
    elo_df = pd.concat(results, ignore_index=True)
    df = df.reset_index(drop=True)
    df['winner_elo'] = elo_df['winner_elo']
    df['loser_elo'] = elo_df['loser_elo']
    return df

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating advanced features...")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    
    numeric_cols = ['winner_elo', 'loser_elo', 'winner_age', 'winner_ht', 'loser_age', 'loser_ht']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    print("Calculating recent form...")
    df['winner_recent_form'] = df.groupby('winner_name')['winner_elo'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['loser_recent_form'] = df.groupby('loser_name')['loser_elo'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    print("Calculating surface win rates...")
    df['winner_surface_win_rate'] = df.groupby(['winner_name', 'surface'])['winner_elo'].transform(
        lambda x: (x > x.shift(1)).mean()
    )
    df['loser_surface_win_rate'] = df.groupby(['loser_name', 'surface'])['loser_elo'].transform(
        lambda x: (x > x.shift(1)).mean()
    )
    
    print("Calculating head-to-head and tournament stats...")
    df['h2h_matches'] = df.groupby(['winner_name', 'loser_name']).cumcount()
    df['winner_tourney_wins'] = df.groupby(['winner_name', 'tourney_name']).cumcount()
    df['loser_tourney_wins'] = df.groupby(['loser_name', 'tourney_name']).cumcount()
    
    print("Calculating days since last match...")
    df['winner_days_since_last_match'] = df.groupby('winner_name')['tourney_date'].diff().dt.days
    df['loser_days_since_last_match'] = df.groupby('loser_name')['tourney_date'].diff().dt.days
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df

def train_ensemble_model(X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
    print("Training ensemble model...")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=15,
        learning_rate=0.1,
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=5,
        tol=1e-4
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=15,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    
    print("Training Gradient Boosting...")
    gb.fit(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)
    return ensemble

def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    print("Optimizing hyperparameters...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'learning_rate': [0.1]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def monte_carlo_simulation(model: VotingClassifier, X: np.ndarray, n_simulations: int = 1000) -> np.ndarray:
    results = []
    for _ in range(n_simulations):
        preds = model.predict_proba(X)
        results.append(preds)
    results = np.array(results)
    win_probs = results.mean(axis=0)
    return win_probs

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Loading data...")
    data_dir = 'TML-Database/'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files")

    df_list = []
    for f in tqdm(csv_files, desc="Loading CSV files"):
        df_list.append(pd.read_csv(os.path.join(data_dir, f)))

    print("Concatenating data...")
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total matches loaded: {len(df)}")

    print("Preprocessing data...")
    df = df.dropna(subset=['winner_name', 'loser_name', 'winner_age', 'winner_ht', 'loser_age', 'loser_ht', 'surface'])
    print(f"Matches after dropping missing values: {len(df)}")

    df = calculate_elo(df)
    df = create_advanced_features(df)

    print("Preparing features...")
    numeric_columns = [
        'winner_elo', 'loser_elo', 'winner_age', 'winner_ht', 'loser_age', 'loser_ht',
        'winner_recent_form', 'loser_recent_form', 'winner_surface_win_rate', 'loser_surface_win_rate',
        'h2h_matches', 'winner_tourney_wins', 'loser_tourney_wins',
        'winner_days_since_last_match', 'loser_days_since_last_match'
    ]

    for col in numeric_columns:
        df[col] = df[col].astype(float)

    cat_features = ['surface']
    X_cat = df[cat_features]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_cat)

    X_num = df[numeric_columns].values
    X_cat_encoded = encoder.transform(X_cat)
    X = np.hstack([X_num, X_cat_encoded])
    y = (df['winner_elo'] > df['loser_elo']).astype(int)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_ensemble_model(X_train, y_train)

    print("Starting hyperparameter optimization...")
    best_params = optimize_hyperparameters(X_train, y_train)
    print("Best hyperparameters:", best_params)

    print("Saving models and encoders...")
    with open('models/tennis_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(numeric_columns + list(encoder.get_feature_names_out(cat_features)), f)

    print("Model and encoders saved successfully!")
    print(f"Trained on {len(df)} matches")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 