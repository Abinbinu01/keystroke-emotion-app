from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

EMOTION_LABELS = ['Happy', 'Sad', 'Calm', 'Stressed']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'No features received'}), 400

    features = data['features']

    feature_vector = np.array([
        features.get('typing_speed_wpm', 0.0),
        features.get('avg_key_interval_ms', 0.0),
        features.get('avg_pause_ms', 0.0),
        features.get('num_key_events', 0.0)
    ], dtype=float)

    wpm = feature_vector[0]
    interval = feature_vector[1]
    pause = feature_vector[2]

    scores = {'Happy': 0.0, 'Sad': 0.0, 'Calm': 0.0, 'Stressed': 0.0}

    # Happy
    if wpm > 45 and interval < 130 and pause < 350:
        scores['Happy'] = 0.9
    elif wpm > 40:
        scores['Happy'] = 0.6

    # Sad
    if wpm < 25 and pause > 800:
        scores['Sad'] = max(scores['Sad'], 0.9)
    elif wpm < 28:
        scores['Sad'] = max(scores['Sad'], 0.6)

    # Stressed
    if wpm > 45 and (interval > 200 or pause > 700):
        scores['Stressed'] = max(scores['Stressed'], 0.9)
    elif interval > 220:
        scores['Stressed'] = max(scores['Stressed'], 0.6)

    # Calm
    if 28 <= wpm <= 40 and 120 <= interval <= 180 and pause < 600:
        scores['Calm'] = max(scores['Calm'], 0.9)
    elif 28 <= wpm <= 42:
        scores['Calm'] = max(scores['Calm'], 0.6)

    total = sum(scores.values())
    if total == 0:
        final_emotion = 'Calm'
        final_confidence = 1.0
    else:
        for k in scores:
            scores[k] /= total
        final_emotion = max(scores, key=scores.get)
        final_confidence = scores[final_emotion]

    return jsonify({
        'emotion': final_emotion,
        'confidence': float(final_confidence)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
