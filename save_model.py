import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

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

if not os.path.exists('models'):
    os.makedirs('models')

data_dir = 'TML-Database/'
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
df_list = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['winner_name', 'loser_name', 'winner_age', 'winner_ht', 'loser_age', 'loser_ht', 'surface'])
df = calculate_elo(df)
numeric_columns = ['winner_elo', 'loser_elo', 'winner_age', 'winner_ht', 'loser_age', 'loser_ht']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_columns)
cat_features = ['surface']
X_cat = df[cat_features]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X_cat)
X_num = df[numeric_columns].values
X_cat_encoded = encoder.transform(X_cat)
X = np.hstack([X_num, X_cat_encoded])
y = (df['winner_elo'] > df['loser_elo']).astype(int)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
with open('models/tennis_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("Model and encoder saved successfully!")
print(f"Trained on {len(df)} matches") 