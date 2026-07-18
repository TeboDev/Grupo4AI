import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODELS = {
    'Regresión Logística': os.path.join(BASE_DIR, 'Diabetes', '2. Regresion Logistica', 'logistic_regression_diabetes.pkl'),
    'K-Nearest Neighbors (KNN)': os.path.join(BASE_DIR, 'Diabetes', '3. Knn', 'knn_model_diabetes.pkl'),
    'Naive Bayes': os.path.join(BASE_DIR, 'Diabetes', '6. Naive Bayes', 'NB_model_diabetes.pkl'),
    'Support Vector Machine (SVM)': os.path.join(BASE_DIR, 'Diabetes', '7. SVM', 'SVM_model_diabetes.pkl'),
    'Decision Tree': os.path.join(BASE_DIR, 'Diabetes', '4. Decision Tree', 'decision_tree_diabetes.pkl'),
    'Random Forest': os.path.join(BASE_DIR, 'Diabetes', '5. Random Forest', 'random_forest_diabetes.pkl'),
    'Red Neuronal Artificial (RNA)': os.path.join(BASE_DIR, 'Diabetes', '8. RNA', 'RNA_model_diabetes.pkl')
}

# Min and Max for each feature based on original dataset to apply MinMaxScaler
SCALER_PARAMS = {
    'Glucosa_Almuerzo': {'min': 15.0, 'max': 452.0},
    'Glucosa_Cena': {'min': 28.0, 'max': 450.0},
    'Glucosa_Desayuno': {'min': 23.0, 'max': 450.0},
    'Insulina_NPH': {'min': 1.0, 'max': 388.0},
    'Insulina_Regular': {'min': 0.5, 'max': 115.33333333333333}
}

def scale_value(feature, value):
    min_val = SCALER_PARAMS[feature]['min']
    max_val = SCALER_PARAMS[feature]['max']
    # If the value is outside the training bounds, we just clip it or let it scale beyond 0-1
    return (value - min_val) / (max_val - min_val)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=list(MODELS.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get model selection
        model_name = request.form.get('model_select')
        
        # 2. Get inputs
        glucosa_almuerzo = float(request.form.get('Glucosa_Almuerzo'))
        glucosa_cena = float(request.form.get('Glucosa_Cena'))
        glucosa_desayuno = float(request.form.get('Glucosa_Desayuno'))
        insulina_nph = float(request.form.get('Insulina_NPH'))
        insulina_regular = float(request.form.get('Insulina_Regular'))
        
        # 3. Scale inputs to match training data (MinMaxScaler)
        X_input = pd.DataFrame([{
            'Glucosa_Almuerzo': scale_value('Glucosa_Almuerzo', glucosa_almuerzo),
            'Glucosa_Cena': scale_value('Glucosa_Cena', glucosa_cena),
            'Glucosa_Desayuno': scale_value('Glucosa_Desayuno', glucosa_desayuno),
            'Insulina_NPH': scale_value('Insulina_NPH', insulina_nph),
            'Insulina_Regular': scale_value('Insulina_Regular', insulina_regular)
        }])
        
        # 4. Load the selected model
        model_path = MODELS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            return render_template('index.html', models=list(MODELS.keys()), error="El modelo seleccionado no se encuentra disponible.")
        
        model = joblib.load(model_path)
        
        # 5. Predict
        # For some models we might want probability, but let's stick to binary classification
        prediction = model.predict(X_input)[0]
        
        result_text = "Alto Riesgo de Hipoglucemia" if prediction == 1 else "Bajo Riesgo de Hipoglucemia"
        result_class = "danger" if prediction == 1 else "success"
        
        return render_template('index.html', models=list(MODELS.keys()), 
                               result=result_text, result_class=result_class, 
                               model_used=model_name)
        
    except Exception as e:
        return render_template('index.html', models=list(MODELS.keys()), error=str(e))

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
