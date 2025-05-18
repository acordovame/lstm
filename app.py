from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo y el scaler
modelo = load_model("modelo_lstm_12horas.h5")
scaler = joblib.load("scaler.pkl")

# Configuración de predicción
INPUT_STEPS = 1
OUTPUT_STEPS = 12

# Características usadas como entrada
features = ['year', 'month', 'day', 'hour', 'visibilidad', 'cantidad_capas', 'temperatura', 'punto_rocio', 'qnh']

# Columnas objetivo
target_columns = ['temperatura', 'visibilidad', 'punto_rocio', 'qnh']  # 4 columnas

@app.route('/')
def index():
    return "✅ API para predicción del clima con LSTM está activa."

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.get_json(force=True)
        entrada = np.array(data['input'])

        if entrada.shape != (1, INPUT_STEPS, len(features)):
            return jsonify({'error': f'Forma inválida: {entrada.shape}. Se esperaba (1, {INPUT_STEPS}, {len(features)}).'}), 400

        # Escalar
        entrada_2d = entrada.reshape(-1, len(features))
        entrada_scaled = scaler.transform(entrada_2d).reshape(1, INPUT_STEPS, len(features))

        # Predicción
        pred_scaled = modelo.predict(entrada_scaled)

        # Como solo tenemos salida de las columnas objetivo, NO usamos scaler.inverse_transform()
        # a menos que hayas entrenado el scaler solo con target_columns (lo cual es poco común)

        # Si los valores predichos ya están en el mismo espacio que target_columns (ej. normalizados por separado)
        # debes aplicar un inverso por variable, pero aquí asumimos que no están normalizados con el mismo scaler general

        # Si el scaler fue entrenado con TODAS las features y target_columns están incluidas,
        # debes reconstruir un array completo para aplicar inverse_transform correctamente
        dummy_input = np.zeros((OUTPUT_STEPS, len(features)))
        target_indices = [features.index(col) for col in target_columns]
        for i, idx in enumerate(target_indices):
            dummy_input[:, idx] = pred_scaled[:, i]

        pred_inv = scaler.inverse_transform(dummy_input)
        pred_result = pred_inv[:, target_indices]  # Extraer solo las columnas que nos interesan

        # Convertir a diccionario JSON-friendly
        resultados = {
            'temperatura': pred_result[:, 0].tolist(),
            'visibilidad': pred_result[:, 1].tolist(),
            'punto_rocio': pred_result[:, 2].tolist(),
            'qnh': pred_result[:, 3].tolist()
        }

        return jsonify({'prediccion': resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
