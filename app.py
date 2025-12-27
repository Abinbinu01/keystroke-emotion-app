from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# --------------------------------------------------------
# Flask app
# --------------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

MODEL_PATH = 'model.h5'
SCALER_PATH = 'scaler.pkl'
EMOTION_LABELS = ['Happy', 'Sad', 'Calm', 'Stressed']

model = None
scaler = None


# --------------------------------------------------------
# Load model + scaler (once at startup)
# --------------------------------------------------------
def load_artifacts():
    global model, scaler

    print("Loading artifacts...")

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    else:
        print(f"{MODEL_PATH} not found. Train the model first.")
        model = None

    if os.path.exists(SCALER_PATH):
        print(f"Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"{SCALER_PATH} not found. Train the model first.")
        scaler = None


load_artifacts()


# --------------------------------------------------------
# Routes
# --------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("/predict called")

    if model is None or scaler is None:
        return jsonify({
            'error': 'Model or scaler not loaded. Train the model first.'
        }), 500

    data = request.get_json()
    print("Raw JSON:", data)

    if not data or 'features' not in data:
        return jsonify({'error': 'No features received'}), 400

    features = data.get('features', {})
    print("Features:", features)

    # Build feature vector
    feature_vector = np.array([
        features.get('typing_speed_wpm', 0.0),
        features.get('avg_key_interval_ms', 0.0),
        features.get('avg_pause_ms', 0.0),
        features.get('num_key_events', 0.0)
    ], dtype=float)

    # ------------------ model prediction ------------------
    feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
    sequence = np.expand_dims(feature_vector_scaled, axis=1)

    probs = model.predict(sequence, verbose=0)[0]
    model_idx = int(np.argmax(probs))
    model_emotion = EMOTION_LABELS[model_idx]
    model_confidence = float(probs[model_idx])

    # ---------------- rule-based overlay ------------------
    scores = {'Happy': 0.0, 'Sad': 0.0, 'Calm': 0.0, 'Stressed': 0.0}

    wpm = feature_vector[0]
    interval = feature_vector[1]
    pause = feature_vector[2]

    # Happy: fast, smooth
    if wpm > 45 and interval < 130 and pause < 350:
        scores['Happy'] = 0.9
    elif wpm > 40:
        scores['Happy'] = 0.6

    # Sad: slow, long pauses
    if wpm < 25 and pause > 800:
        scores['Sad'] = max(scores['Sad'], 0.9)
    elif wpm < 28:
        scores['Sad'] = max(scores['Sad'], 0.6)

    # Stressed: fast but irregular
    if wpm > 45 and (interval > 200 or pause > 700):
        scores['Stressed'] = max(scores['Stressed'], 0.9)
    elif interval > 220:
        scores['Stressed'] = max(scores['Stressed'], 0.6)

    # Calm: medium everything
    if 28 <= wpm <= 40 and 120 <= interval <= 180 and pause < 600:
        scores['Calm'] = max(scores['Calm'], 0.9)
    elif 28 <= wpm <= 42:
        scores['Calm'] = max(scores['Calm'], 0.6)

    total = sum(scores.values())
    if total == 0:
        final_emotion = model_emotion
        final_confidence = model_confidence
    else:
        for k in scores:
            scores[k] /= total
        final_emotion = max(scores, key=scores.get)
        final_confidence = scores[final_emotion]

    print(
        f"Model: {model_emotion} ({model_confidence:.2f}) | "
        f"Final: {final_emotion} ({final_confidence:.2f})"
    )

    return jsonify({
        'emotion': final_emotion,
        'confidence': float(final_confidence),
        'model_emotion': model_emotion,
        'model_confidence': model_confidence
    })


# --------------------------------------------------------
# Main (for local run; Render uses gunicorn app:app)
# --------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
