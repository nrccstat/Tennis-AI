# TennisAI: ATP Match Prediction

This project is a machine learning pipeline and web app for predicting ATP tennis match outcomes and simulating tournament brackets.

## Features
- Uses historical ATP match data for feature engineering and model training
- Calculates Elo ratings for all players
- Predicts match outcomes and win probabilities
- Interactive Streamlit UI for user-friendly predictions
- Modular code for easy extension and experimentation

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
