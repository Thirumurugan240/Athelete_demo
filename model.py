import pickle
import numpy as np

# Load trained model
with open("xgboost_injury_risk.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_injury_risk(player_data):
    predictions = []
    for player in player_data:
        features = np.array([[player["Minutes_Played"], player["Sprint_Count"]]])
        features_scaled = scaler.transform(features)
        risk_score = model.predict_proba(features_scaled)[:, 1][0]
        playability = max(0, 100 - (risk_score * 100))  
        predicted_risk = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"

        predictions.append({
            "Player_ID": player["Player_ID"],
            "Risk_Score": round(risk_score, 2),
            "Playability": round(playability, 2),
            "Predicted_Risk": predicted_risk
        })
    return predictions
