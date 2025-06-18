# TennisAI: ATP Match Prediction

This project is a machine learning pipeline and web app for predicting ATP tennis match outcomes and simulating tournament brackets.

## Features
- Uses historical ATP match data for feature engineering and model training
- Calculates Elo ratings for all players
- Predicts match outcomes and win probabilities
- Interactive Streamlit UI for user-friendly predictions
- Modular code for easy extension and experimentation

## Methodology
This project uses a comprehensive machine learning pipeline for ATP tennis match prediction. The methodology includes:

### 1. Data Collection & Preprocessing
- Historical ATP match data is sourced from the TML-Database.
- Data is cleaned and filtered to remove matches with missing critical information (e.g., player names, ages, heights, surface).

### 2. Feature Engineering
- **Elo Ratings:** Dynamic Elo ratings are calculated for each player, updating after every match to reflect recent performance.
- **Recent Form:** Rolling average of a player's Elo over the last 5 matches to capture momentum.
- **Surface Win Rate:** Player's win rate on each surface type (Hard, Clay, Grass, Carpet) is computed.
- **Head-to-Head (H2H):** Number of previous matches between the two players.
- **Tournament Wins:** Number of times a player has won matches in the current tournament.
- **Days Since Last Match:** Time since each player's last match, to account for rest/fatigue.
- All numeric features are filled with median values where missing.

### 3. Feature Encoding
- **Categorical Encoding:** The 'surface' feature is one-hot encoded using `OneHotEncoder` to allow the model to learn surface-specific effects.
- **Feature Vector:** The final feature vector for each match includes 15 numeric features and the one-hot encoded surface features, matching the training pipeline.

### 4. Model Training
- **Ensemble Model:** A soft-voting ensemble is trained, combining:
  - RandomForestClassifier
  - GradientBoostingClassifier
  - XGBoostClassifier
- The ensemble is trained on the engineered features to predict the probability that the first player wins.
- **Hyperparameter Optimization:** Grid search is used to tune model hyperparameters for best performance.

### 5. Model Saving
- The trained ensemble model, encoder, and feature column order are saved to the `models/` directory for reproducible inference.

### 6. Prediction & UI
- The Streamlit UI allows users to select players and surface, and computes all required features (using defaults where historical data is missing).
- The model outputs win probabilities, match insights, and player statistics.

## Data Source
All data analysis in this project uses the [TML-Database](https://github.com/Tennismylife/TML-Database), a complete and live-updated database of ATP tournament matches. Huge thanks to the TML-Database authors for their work and for making this resource available.

## Setup
1. Clone this repository and the [TML-Database](https://github.com/Tennismylife/TML-Database) into the `TML-Database/` directory.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python save_model.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Attribution
- Data: [TML-Database](https://github.com/Tennismylife/TML-Database) by Tennismylife and contributors
- Project by: Narasimha Cittarusu
