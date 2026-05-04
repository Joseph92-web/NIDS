# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Paths
MODEL_PATH = 'models/model.pkl'
FEATURES_PATH = 'models/features.pkl'
ENCODERS_PATH = 'models/preprocessors/label_encoders.pkl'
SCALER_PATH = 'models/preprocessors/scaler.pkl'

# Lazy loading
model = None
expected_columns = None
label_encoders = None
scaler = None

def load_artifacts():
    global model, expected_columns, label_encoders, scaler
    try:
        if model is None:
            model = joblib.load(MODEL_PATH)
        if expected_columns is None:
            expected_columns = joblib.load(FEATURES_PATH)
        if label_encoders is None:
            label_encoders = joblib.load(ENCODERS_PATH)
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

# Class mapping
INT_TO_CLASS = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
SEVERITY = {'Normal': 'Low', 'DoS': 'High', 'Probe': 'Medium', 'R2L': 'Critical', 'U2R': 'Critical'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect_form():
    return render_template('detect.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not load_artifacts():
        return jsonify({'error': 'Model not loaded. Please train first.'}), 500
    
    try:
        data = request.get_json()
        # Convert to DataFrame with single row
        df = pd.DataFrame([data])
        
        # Ensure all expected columns exist, fill missing with 0
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select and order
        df = df[expected_columns]
        
        # Apply label encoders for categorical columns
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                # Replace unseen labels with most frequent? Here we default to first class if unknown
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # If unseen label appears, map to 0 (or could use a default)
                    df[col] = 0
        
        # Scale
        X_scaled = scaler.transform(df.values)
        
        # Predict
        pred_int = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]
        
        pred_class = INT_TO_CLASS.get(pred_int, 'Unknown')
        confidence = float(max(pred_proba) * 100)
        
        return jsonify({
            'prediction': int(pred_int),
            'class': pred_class,
            'confidence': round(confidence, 2),
            'severity': SEVERITY.get(pred_class, 'Unknown'),
            'probabilities': {
                'Normal': round(pred_proba[0] * 100, 2),
                'DoS': round(pred_proba[1] * 100, 2),
                'Probe': round(pred_proba[2] * 100, 2),
                'R2L': round(pred_proba[3] * 100, 2),
                'U2R': round(pred_proba[4] * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    if not load_artifacts():
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return jsonify({'error': f'Missing columns: {list(missing_cols)}'}), 400
        
        df = df[expected_columns]
        
        # Encode categorical
        for col in ['protocol_type', 'service', 'flag']:
            if col in label_encoders:
                # Map each value, replace unknowns with 0
                df[col] = df[col].astype(str).apply(
                    lambda x: label_encoders[col].transform([x])[0] 
                    if x in label_encoders[col].classes_ else 0
                )
        
        # Scale
        X_scaled = scaler.transform(df.values)
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'row': i,
                'prediction': int(pred),
                'class': INT_TO_CLASS.get(int(pred), 'Unknown'),
                'confidence': float(max(probabilities[i]) * 100)
            })
        
        attack_count = sum(predictions != 0)
        
        return jsonify({
            'total': len(df),
            'attacks': int(attack_count),
            'normal': len(df) - int(attack_count),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting NIDS Flask Application...")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Features file exists: {os.path.exists(FEATURES_PATH)}")
    print("Access the dashboard at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)