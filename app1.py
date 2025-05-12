
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and symptoms
model = joblib.load('disease_predictor.joblib')
with open('symptoms.txt', 'r') as f:
    symptoms = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get('symptoms', [])
    
    # Create input vector
    input_vector = np.zeros(len(symptoms))
    for symptom in user_symptoms:
        if symptom in symptoms:
            idx = symptoms.index(symptom)
            input_vector[idx] = 1
    
    # Make prediction
    probabilities = model.predict_proba([input_vector])[0]
    top5_idx = np.argsort(probabilities)[-5:][::-1]
    top5_diseases = model.classes_[top5_idx]
    top5_probs = probabilities[top5_idx]
    
    results = [{'disease': str(d), 'probability': float(p)} 
              for d, p in zip(top5_diseases, top5_probs)]
    
    return jsonify({'predictions': results})

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': symptoms})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    