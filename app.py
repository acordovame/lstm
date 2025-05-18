from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo y scaler
modelo = load_model("modelo.keras")
scaler = joblib.load("scaler.pkl")

# Configuración
INPUT_STEPS = 1
OUTPUT_STEPS = 12

# Deben coincidir con lo usado al entrenar
features = ['year','month','day','hour', 'visibilidad', 'cantidad_capas' ,'temperatura', 'punto_rocio', 'qnh']
target_columns = ['temperatura','visibilidad', 'punto_rocio', 'qnh']

# Inversa del escalado (como hiciste en entrenamiento)
def inverse_transform_column(pred_column, col_index):
    dummy = np.zeros((pred_column.shape[0], len(features)))
    dummy[:, col_index] = pred_column[:, 0]
    return scaler.inverse_transform(dummy)[:, col_index]

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

        # Escalar entrada
        entrada_2d = entrada.reshape(-1, len(features))
        entrada_scaled = scaler.transform(entrada_2d).reshape(1, INPUT_STEPS, len(features))

        # Predicción
        pred_scaled = modelo.predict(entrada_scaled)

        # Desnormalización (por columna)
        resultados = {}
        for i, var in enumerate(target_columns):
            col_index = features.index(var)
            pred_column = pred_scaled[:, i].reshape(-1, 1)
            desnormalizado = inverse_transform_column(pred_column, col_index)
            resultados[var] = desnormalizado.tolist()

        return jsonify({'prediccion': resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
