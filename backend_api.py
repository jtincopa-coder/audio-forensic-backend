import sys
import os
import tempfile
import urllib.request
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
import joblib
from scipy.io import wavfile

# === CONFIGURACIÓN DE MODELOS ===
# Reemplaza estos IDs con los tuyos de Google Drive
MODEL_4CL_URL = "https://drive.google.com/uc?export=download&id=1xHZjrPcfISGkZMzwlArmGr1abSNNWe8X"
MODEL_3CL_URL = "https://drive.google.com/uc?export=download&id=1HolhHoEfoyqoI0AHlv4t-yeggXBFpgzV"
LE_4CL_URL = "https://drive.google.com/uc?export=download&id=1na5YrHC-8MIAn-tKQC3OWgax1pUq5OOU"
LE_3CL_URL = "https://drive.google.com/uc?export=download&id=17s2tSJVDfgPgMQVrXC-eTzpjjzb6NVHF"

def download_if_missing(url, path):
    """Descarga un archivo si no existe localmente"""
    if not os.path.exists(path):
        print(f"📥 Descargando {os.path.basename(path)}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"✅ {os.path.basename(path)} descargado.")
        except Exception as e:
            print(f"❌ Error al descargar {os.path.basename(path)}: {e}")
            raise

# === Ruta segura para PyInstaller o entornos normales ===
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# === Descargar modelos si es necesario ===
base_dir = os.path.dirname(os.path.abspath(__file__))
model_4cl_path = os.path.join(base_dir, "model_cnn_final_combined.h5")
model_3cl_path = os.path.join(base_dir, "model_lethality_with_aug.h5")
le_4cl_path = os.path.join(base_dir, "label_encoder.pkl")
le_3cl_path = os.path.join(base_dir, "label_encoder_lethality.pkl")

# Solo intentar descargar si no estamos en modo empaquetado (PyInstaller)
if not getattr(sys, 'frozen', False):
    download_if_missing(MODEL_4CL_URL, model_4cl_path)
    download_if_missing(MODEL_3CL_URL, model_3cl_path)
    download_if_missing(LE_4CL_URL, le_4cl_path)
    download_if_missing(LE_3CL_URL, le_3cl_path)

# === Cargar modelos ===
print("🧠 Cargando modelos...")
model_4cl = keras.models.load_model(model_4cl_path)
model_3cl = keras.models.load_model(model_3cl_path)
le_4cl = joblib.load(le_4cl_path)
le_3cl = joblib.load(le_3cl_path)
class_names_4cl = ["Arma_Corta", "Escopeta_Perdigones", "Lacrimogena", "Rifle_Fusil"]
class_names_3cl = ["No Letal", "Letalidad Intermedia", "Letal"]
print("✅ Modelos cargados.")

# === Cargar preprocessing ===
sys.path.insert(0, base_dir)
from preprocessing import extract_features

app = Flask(__name__)

def load_wav_safe(path):
    """Carga WAV sin dependencias externas"""
    sr, y = wavfile.read(path)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # Convertir a mono
    return y, sr

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se recibió archivo de audio'}), 400

        audio_file = request.files['audio']
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', start_time + 1.0))

        if end_time <= start_time:
            return jsonify({'error': 'El tiempo de fin debe ser mayor que el de inicio'}), 400

        # Guardar temporalmente
        temp_dir = tempfile.gettempdir()
        temp_input_path = os.path.join(temp_dir, f"input_{os.urandom(4).hex()}.wav")
        audio_file.save(temp_input_path)

        # Cargar y recortar
        y, sr = load_wav_safe(temp_input_path)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_seg = y[start_sample:end_sample]

        # Guardar segmento
        segment_path = os.path.join(temp_dir, f"seg_{os.urandom(4).hex()}.wav")
        wavfile.write(segment_path, sr, (y_seg * 32767).astype(np.int16))

        # Extraer características
        features = extract_features(segment_path)
        if features.shape[1] == 153:
            features = features[:, 13:13+128, :]
        features = features.astype(np.float32)

        # Predicciones
        pred_4 = model_4cl.predict(features[np.newaxis], verbose=0)[0]
        pred_3 = model_3cl.predict(features[np.newaxis], verbose=0)[0]

        weapon_idx = int(np.argmax(pred_4))
        lethality_idx = int(np.argmax(pred_3))

        result = {
            'success': True,
            'weapon': {'name': class_names_4cl[weapon_idx], 'confidence': float(pred_4[weapon_idx] * 100)},
            'lethality': {'class': class_names_3cl[lethality_idx], 'confidence': float(pred_3[lethality_idx] * 100)}
        }

        # Limpiar
        for p in [temp_input_path, segment_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except:
                pass

        return jsonify(result), 200

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'models_loaded': True,
        'model_4cl_classes': class_names_4cl,
        'model_3cl_classes': class_names_3cl
    }), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
