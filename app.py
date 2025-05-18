from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo LSTM y el scaler
modelo = load_model("modelo_lstm_12horas.h5")  # Actualiza el nombre del modelo si es necesario
scaler = joblib.load("scaler.pkl")  # Asegúrate de que el scaler haya sido guardado previamente con joblib

# Configuración de parámetros de predicción (se pueden ajustar según necesidad)
INPUT_STEPS = 36       # Número de pasos de entrada (registros anteriores)
PREDICT_SHIFT = 1     # Número de pasos hacia el futuro (predicción 12 horas más tarde)
OUTPUT_STEPS = 12       # Número de pasos de salida (por defecto 1, si deseas predecir varios valores futuros ajusta este valor)

# Definir las características y las columnas de salida que se predicen
features = ['year','month','day','hour','dir_viento','vel_viento','visibilidad','temperatura', 'punto_rocio', 'humedad_r', 'qnh','qfe','precipitacion']
target_columns = ['temperatura', 'humedad_r', 'qnh']

@app.route('/')
def index():
    return "✅ API para predicción del clima con LSTM está activa."

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Obtener datos del cuerpo de la solicitud
        data = request.get_json(force=True)
        
        # Asegurarse de que la entrada sea un arreglo numpy de la forma correcta
        entrada = np.array(data['input'])  # Entrada esperada: (1, 60, 13) - 60 registros, 13 características

        # Verificar que la forma sea correcta
        if entrada.shape != (1, INPUT_STEPS, len(features)):
            return jsonify({'error': f'Forma inválida: {entrada.shape}. Se esperaba (1, {INPUT_STEPS}, {len(features)}).'}), 400

        # Escalar la entrada utilizando el mismo escalador que se usó durante el entrenamiento
        entrada_2d = entrada.reshape(-1, len(features))  # Aplanar para el escalado: (60*13, 1)
        entrada_scaled = scaler.transform(entrada_2d)
        entrada_scaled = entrada_scaled.reshape(1, INPUT_STEPS, len(features))  # Volver a dar la forma (1, 60, 13)

        # Realizar la predicción
        pred_scaled = modelo.predict(entrada_scaled)

        # Desnormalizar la predicción para obtener los valores reales
        # El modelo devuelve una salida para cada paso (OUTPUT_STEPS pasos, 3 variables)
        pred = pred_scaled.reshape(-1, len(target_columns))  # Aplanar la salida: (OUTPUT_STEPS, 3)

        # Usar el escalador para revertir la normalización (restaurar las 3 variables)
        pred_inv = scaler.inverse_transform(pred)

        # Organizar los resultados en formato JSON
        resultados = {
            'temperatura': pred_inv[:, 0].tolist(),  # Temperatura
            'humedad_r': pred_inv[:, 1].tolist(),  # Humedad relativa
            'qnh': pred_inv[:, 2].tolist()  # Presión atmosférica
        }

        return jsonify({'prediccion': resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Usar un puerto de entorno para despliegue en plataformas como Render
    app.run(host="0.0.0.0", port=port)
