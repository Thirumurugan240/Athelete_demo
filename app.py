import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from video_processor import process_video
import xgboost as xgb
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load model and scaler
model_path = "xgboost_injury_risk.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Error: Model or scaler file not found!")

with open(model_path, "rb") as f:
    injury_model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

if not isinstance(scaler, StandardScaler):
    raise TypeError("Error: scaler.pkl is not a valid StandardScaler object!")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No file uploaded", 400

        file = request.files["video"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        extracted_players = process_video(file_path)
        if not extracted_players:
            return "No players detected in the video.", 500

        for player in extracted_players:
            try:
                input_features = np.array([[  
                    player.get("Minutes_Played", 0), 
                    player.get("Sprint_Count", 0),
                    player.get("Fatigue_Index", 0),
                    player.get("Sprint_Intensity", 0),
                    player.get("Composite_Load_Score", 0),
                    player.get("Injury_History", 0)
                ]])

                input_scaled = scaler.transform(input_features)
                risk_score = float(injury_model.predict_proba(input_scaled)[:, 1][0])  # Convert to Python float

                player["Risk_Score"] = round(risk_score, 2)
                player["Predicted_Risk"] = "High" if risk_score > 0.8 else "Low"
                player["Playability"] = max(0, 100 - (risk_score * 100))
                player["Injury_Risk"] = round(risk_score * 100, 2)  

            except Exception as e:
                print(f"Error processing player {player['Player_ID']}: {e}")

        highest_risk_player = max(extracted_players, key=lambda x: x["Risk_Score"], default=None)
        return jsonify({
            "players": extracted_players,
            "highest_risk": highest_risk_player
        })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
