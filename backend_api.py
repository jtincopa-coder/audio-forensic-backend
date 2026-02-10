import sys
import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
import joblib
from scipy.io import wavfile
from pydub import AudioSegment
import io

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# Cargar modelos
print("Cargando modelos...")
model_4cl = keras.models.load_model(resource_path("model_cnn_final_combined.h5"))
model_3cl = keras.models.load_model(resource_path("model_lethality_with_aug.h5"))
class_names_4cl = ["Arma_Corta", "Escopeta_Perdigones", "Lacrimogena", "Rifle_Fusil"]
class_names_3cl = ["No Letal", "Letalidad Intermedia", "Letal"]
print("Modelos cargados.")

# Cargar preprocessing
sys.path.insert(0, os.path.dirname(resource_path("preprocessing.py")))
from preprocessing import extract_features

app = Flask(__name__)

def load_audio_safe(file_path):
    """Carga cualquier formato de audio y lo convierte a numpy array (mono, 16kHz)"""
    try:
        audio = AudioSegment.from_file(file_path)
        # Convertir a mono y 16 kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        # Exportar a bytes en memoria
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        # Cargar con scipy
        sr, y = wavfile.read(wav_bytes)
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        else:
            y = y.astype(np.float32)
        return y, sr
    except Exception as e:
        raise ValueError(f"Error al cargar audio: {e}")

def save_wav_safe(path, sr, y):
    """Guarda WAV sin soundfile"""
    y_norm = np.clip(y, -1.0, 1.0)
    y_int16 = (y_norm * 32767).astype(np.int16)
    wavfile.write(path, sr, y_int16)

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
        y, sr = load_audio_safe(temp_input_path)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_seg = y[start_sample:end_sample]

        # Guardar segmento
        segment_path = os.path.join(temp_dir, f"seg_{os.urandom(4).hex()}.wav")
        save_wav_safe(segment_path, sr, y_seg)

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
        print(f"ERROR: {e}")
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
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
