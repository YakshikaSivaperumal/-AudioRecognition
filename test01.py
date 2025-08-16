import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import pyaudio
import wave
import os

# Model loading function
def load_audio_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to get spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Normalize waveform
def normalize_waveform(waveform):
    waveform = tf.cast(waveform, tf.float32)
    waveform = waveform / tf.reduce_max(tf.abs(waveform))
    return waveform

# Predict audio function
def predict_audio(model, waveform):
    class_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

    # Normalize and process waveform
    waveform = normalize_waveform(waveform)
    waveform = waveform[tf.newaxis, :]  # Add batch dimension
    spectrogram = get_spectrogram(waveform)

    # Make predictions
    predictions = model(spectrogram, training=False)
    class_id = tf.argmax(predictions[0]).numpy()
    confidence = tf.nn.softmax(predictions[0])[class_id].numpy() * 100
    predicted_class = class_names[class_id]

    return predicted_class, confidence

# Record audio using PyAudio
def record_audio(duration=1, sample_rate=16000):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    audio = pyaudio.PyAudio()

    print("Recording...")
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Recording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Combine frames into waveform
    waveform = np.concatenate(frames, axis=0)

    # Ensure waveform is exactly 16000 samples
    if len(waveform) > 16000:
        waveform = waveform[:16000]  # Truncate
    elif len(waveform) < 16000:
        waveform = np.pad(waveform, (0, 16000 - len(waveform)), mode='constant')  # Pad with zeros

    return waveform

# GUI application
class RealTimeSpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Speech Recognition")
        self.root.geometry("600x600")
        self.root.configure(bg="#f0f0f0")  # Background color

        # Load model
        self.model_path = "aud_model.hdf5"  # Path to your model
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Model file not found!")
            self.root.destroy()
            return
        self.model = load_audio_model(self.model_path)

        # UI Elements
        self.label = tk.Label(root, text="Real-Time Speech Recognition", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333333")
        self.label.pack(pady=20)

        self.record_button = tk.Button(root, text="Start Recording", command=self.start_recording,
                                       bg="#0078D7", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.record_button.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333333")
        self.result_label.pack(pady=20)

    def start_recording(self):
        try:
            # Record audio for 2 seconds
            waveform = record_audio(duration=2)
            # Predict using the model
            predicted_class, confidence = predict_audio(self.model, waveform)
            self.result_label.config(text=f"Predicted: {predicted_class} ({confidence:.2f}%)")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeSpeechRecognitionApp(root)
    root.mainloop()
