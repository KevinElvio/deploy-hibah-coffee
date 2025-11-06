from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import os

# =========================
# Inisialisasi Flask App
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Path ke model dan scaler
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_x.joblib')

# =========================
# Load Model & Scaler
# =========================
try:
    print("🔄 Loading model and scaler...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load(SCALER_PATH)
    labels = ["Arabica", "Robusta"]
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model = None
    scaler = None
    labels = ["Unknown"]

# =========================
# Route Utama
# =========================
@app.route('/')
def index():
    return jsonify({"message": "Flask model API is running successfully 🚀"})

# =========================
# Route Prediksi
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly.'}), 500

    data = request.get_json(force=True)
    print("📦 Received data:", data)

    features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Sweetness']

    try:
        # Validasi & konversi input
        input_values = [float(data.get(f)) for f in features]
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    # Skala data
    try:
        scaled_data = scaler.transform([input_values])
        scaled_data = np.expand_dims(scaled_data, axis=1)  # untuk LSTM
    except Exception as e:
        return jsonify({'error': f'Scaling error: {e}'}), 500

    # Prediksi
    try:
        prediction_proba = model.predict(scaled_data)
        predicted_index = int(np.argmax(prediction_proba, axis=1)[0])
        predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"
    except Exception as e:
        return jsonify({'error': f'Model prediction error: {e}'}), 500

    return jsonify({
        'prediction': predicted_label,
        'probabilities': prediction_proba.tolist()
    })

# =========================
# Main Runner
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
