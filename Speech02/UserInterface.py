import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
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
def predict_audio(model, audio_path):
    class_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

    # Load and decode the audio file
    audio_binary = tf.io.read_file(audio_path)
    waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    waveform = tf.squeeze(waveform, axis=-1)

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

# GUI application
class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition")
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
        self.label = tk.Label(root, text="Speech Recognition", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333333")
        self.label.pack(pady=20)

        self.file_label = tk.Label(root, text="No file selected", font=("Arial", 12), bg="#f0f0f0", fg="#555555", wraplength=500)
        self.file_label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Audio File", command=self.select_file,
                                       bg="#0078D7", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.select_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict,
                                        bg="#28A745", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333333")
        self.result_label.pack(pady=20)

        self.audio_path = None

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.audio_path = file_path
            self.file_label.config(text=f"Selected: {os.path.basename(file_path)}")
        else:
            self.file_label.config(text="No file selected")

    def predict(self):
        if not self.audio_path:
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return
        try:
            predicted_class, confidence = predict_audio(self.model, self.audio_path)
            self.result_label.config(text=f"Predicted: {predicted_class} ({confidence:.2f}%)")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()
