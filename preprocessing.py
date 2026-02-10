import librosa
import numpy as np
from scipy.signal import butter, filtfilt

def highpass_filter(y, sr, cutoff=700.0): #filtro pasa alto (mayor a 700 hz)
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, y)

def detect_shot_peaks(y, sr, threshold_factor=0.7): #detectar pico del disparo con umbral
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = threshold_factor * np.max(rms)
    peaks = np.where(rms > threshold)[0]
    if len(peaks) == 0:
        return 0, len(y)
    start_frame = max(0, peaks[0] - 2)
    end_frame = min(len(rms) - 1, peaks[-1] + 2)
    return start_frame * hop_length, min(len(y), (end_frame + 1) * hop_length)

def amplify_segment(y, sr, start_samp, end_samp, gain_db=8.0): #amplificar segmento de audio
    y_enhanced = y.copy()
    gain = 10 ** (gain_db / 20)
    y_enhanced[start_samp:end_samp] *= gain
    return y_enhanced

def extract_features(filepath, target_sr=16000): # extraccion de caracteristicas
    """
    Extrae SOLO el espectrograma de Mel (128 bandas).
    Retorna: array de forma (224, 128, 1)
    """
    y, sr = librosa.load(filepath, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # EXTRAER SOLO MEL (128 bandas)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Ajustar a 224 frames
    if mel_db.shape[1] < 224:
        pad = 224 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :224]
    
    return mel_db.T[..., np.newaxis]  # (224, 128, 1)
