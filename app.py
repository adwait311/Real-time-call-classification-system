from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('call_classifier_model.pkl')

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    if 'text' in data:
        # Preprocess and classify
        processed_text = preprocess_text(data['text'])
        prediction = model.predict([processed_text])
        return jsonify({'category': prediction[0]})
    else:
        return jsonify({'error': 'No text provided'}), 400