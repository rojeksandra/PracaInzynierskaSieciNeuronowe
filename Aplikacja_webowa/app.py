import sys
import io
from flask import Flask, jsonify, request, render_template
import pickle
import os
import numpy as np
import librosa
from tensorflow.keras.models import model_from_json
from pydub import AudioSegment  
from flask import Flask, render_template, request, send_from_directory

# Mapowanie emocji z angielskiego na polski
emotion_map = {
    "Angry": "Złość",
    "Happy": "Szczęście",
    "Neutral": "Neutralność",
    "Sad": "Smutek",
    "Surprise": "Zaskoczenie"
}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Konfiguracja folderu do przesyłania plików
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksymalny rozmiar pliku

# Sprawdzanie, czy folder istnieje, jeśli nie, to tworzenie
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Funkcja konwertująca pliki na format WAV
def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")

# Funkcje do ekstrakcji cech dźwięku
def calculate_zcr(audio_data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def calculate_rmse(audio_data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def calculate_mfcc(audio_data, sample_rate, flatten=True):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    return np.ravel(mfcc.T) if flatten else mfcc.T

def calculate_spectral_bandwidth(audio_data, sample_rate, hop_length=512):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, hop_length=hop_length)
    return np.squeeze(spectral_bandwidth)

def calculate_pitch(audio_data, sample_rate, hop_length=512):
    pitches, _ = librosa.core.piptrack(y=audio_data, sr=sample_rate, hop_length=hop_length)
    pitch_values = np.max(pitches, axis=0)
    pitch_values = pitch_values[pitch_values > 0]  
    return pitch_values if len(pitch_values) > 0 else np.array([0])

def calculate_energy(audio_data, frame_length=2048, hop_length=512):
    energy = np.array([np.sum(np.abs(audio_data[i:i+frame_length]**2)) for i in range(0, len(audio_data), hop_length)])
    return energy

def calculate_tempo(audio_data, sample_rate):
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    return tempo

def calculate_rolloff(audio_data, sample_rate, roll_percent=0.85, hop_length=512):
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, roll_percent=roll_percent, hop_length=hop_length)
    return np.squeeze(rolloff)

# Funkcja do ekstrakcji wszystkich cech z audio
def extract_audio_features(data, sr=16000, frame_length=2048, hop_length=512):
    zcr = calculate_zcr(data, frame_length, hop_length)
    rmse = calculate_rmse(data, frame_length, hop_length)
    mfcc = calculate_mfcc(data, sr)
    spectral_bandwidth = calculate_spectral_bandwidth(data, sr, hop_length)
    pitch = calculate_pitch(data, sr, hop_length)
    energy = calculate_energy(data, frame_length, hop_length)
    tempo = calculate_tempo(data, sr)
    rolloff = calculate_rolloff(data, sr, hop_length=hop_length) 
    
    return np.hstack([zcr, rmse, mfcc, spectral_bandwidth, pitch, energy, tempo, rolloff])

# Funkcja do uzyskiwania cech z pliku audio
def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.76, offset=0.4)
    res = extract_audio_features(d)
    result = np.array(res)
    
    # Dopełnienie wektora, jeśli jest zbyt krótki
    if result.size < 3095:
        result = np.pad(result, (0, 3095 - result.size), mode='constant')
    
    result = np.reshape(result, newshape=(1, 3095))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    
    return final_result

# Funkcja przewidująca emocję z pliku
def prediction(path1):
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    
    # Mapowanie emocji na polski
    english_emotion = y_pred[0][0]
    predicted_emotion = emotion_map.get(english_emotion, "Nieznana emocja")  
    return predicted_emotion

# Wczytywanie modelu i innych plików
try:
    with open('C:/visual_app2_praca_inz/model_55epok/CNN_model_55epoch.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('C:/visual_app2_praca_inz/model_55epok/CNN_model_weights_55epoch.weights.h5')
    
    with open('C:/visual_app2_praca_inz/model_55epok/encoder2_55epok.pickle', 'rb') as f:
        encoder2 = pickle.load(f)
    
    with open('C:/visual_app2_praca_inz/model_55epok/scaler2_55epok.pickle', 'rb') as f:
        scaler2 = pickle.load(f)
    
    load_status = "Wszystkie pliki zostały wczytane poprawnie."
except Exception as e:
    load_status = f"Błąd: {str(e)}."


@app.route("/instruction")
def instruction():
    return render_template('instruction.html') 

@app.route("/record")
def record():
    return render_template('record.html') 

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_emotion = None
    error_message = None
    file_path = None 

    if request.method == "POST":
        if 'file' not in request.files:
            error_message = "Brak pliku w żądaniu"
            return render_template('predict.html', error_message=error_message)
        
        file = request.files['file']
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Konwersja pliku na WAV, jeśli nie jest w tym formacie
            if not filename.endswith('.wav'):
                temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.wav')
                file.save(filepath)
                convert_to_wav(filepath, temp_filepath)
                filepath = temp_filepath
            else:
                file.save(filepath)
            
            file_path = '/static/uploads/' + filename  
            
            try:
                predicted_emotion = prediction(filepath)
                return render_template('predict.html', predicted_emotion=predicted_emotion, file_path=file_path)
            except Exception as e:
                error_message = f"Błąd przy przetwarzaniu pliku: {str(e)}"
                return render_template('predict.html', error_message=error_message, file_path=file_path)

        else:
            error_message = "Proszę załadować plik .wav"
            return render_template('predict.html', error_message=error_message)

    return render_template('predict.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
