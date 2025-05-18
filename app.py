from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo y scaler
modelo = load_model("modelo.keras")  # Actualizar nombre del modelo si es necesario
scaler = joblib.load("scaler.pkl")  # Escalador para las entradas

@app.route('/')
def index():
    return "✅ API para predicción del clima con LSTM está activa."

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Obtener datos del cuerpo de la solicitud
        data = request.get_json(force=True)
        
        # Asegurarse de que la entrada sea un arreglo numpy de la forma correcta
        entrada = np.array(data['input'])  # Entrada esperada: (1, 36, 20) - 36 registros, 20 características

        # Verificar que la forma sea correcta
        if entrada.shape != (1, 36, 20):
            return jsonify({'error': f'Forma inválida: {entrada.shape}. Se esperaba (1, 36, 20).'}), 400

        # Escalar la entrada utilizando el mismo escalador que se usó durante el entrenamiento
        entrada_2d = entrada.reshape(-1, 20)  # Aplanar para el escalado: (36*20, 1)
        entrada_scaled = scaler.transform(entrada_2d)
        entrada_scaled = entrada_scaled.reshape(1, 36, 20)  # Volver a dar la forma (1, 36, 20)

        # Realizar la predicción
        pred_scaled = modelo.predict(entrada_scaled)

        # Desnormalizar la predicción para obtener los valores reales
        # El modelo devuelve una salida para cada paso (12 pasos, 3 variables)
        pred = pred_scaled.reshape(-1, 3)  # Aplanar la salida: (12, 3)

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
    port = int(os.environ.get("PORT", 5000))  # Requiere esto para Render o entorno de despliegue
    app.run(host="0.0.0.0", port=port)