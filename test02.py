import tensorflow as tf
import numpy as np
import pyaudio
import tkinter as tk
from tkinter import messagebox

# Function to normalize the waveform
def normalize_waveform(waveform):
    waveform = tf.cast(waveform, tf.float32)
    waveform = waveform / tf.reduce_max(tf.abs(waveform) + 1e-6)  # Avoid division by zero
    return waveform

# Function to generate spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform,
        frame_length=400,  # 25 ms window at 16kHz
        frame_step=160     # 10 ms step
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # Add channel dimension
    return spectrogram

# Function to load the model
def load_audio_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to record real-time audio
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

# Function to predict audio class
def predict_audio(model, waveform):
    class_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']  # Update as per your model
    waveform = normalize_waveform(waveform)

    # Add batch dimension
    waveform = waveform[tf.newaxis, ...]

    # Generate spectrogram
    spectrogram = get_spectrogram(waveform)

    # Ensure correct input shape
    spectrogram = tf.image.resize(spectrogram, [124, 129])

    # Make prediction
    predictions = model(spectrogram, training=False)
    probabilities = tf.nn.softmax(predictions[0])
    predicted_class = class_names[tf.argmax(probabilities)]
    confidence = tf.reduce_max(probabilities) * 100

    return predicted_class, confidence.numpy()

# GUI Implementation
def run_gui(model):
    def start_prediction():
        try:
            waveform = record_audio(duration=1)
            predicted_class, confidence = predict_audio(model, waveform)

            if confidence > 70:  # Confidence threshold
                result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f}%)", fg="green")
            else:
                result_label.config(text="Prediction confidence too low. Try again.", fg="red")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # GUI Window
    root = tk.Tk()
    root.title("Real-Time Speech Recognition")
    root.geometry("600x600")
    root.configure(bg="#e6f7ff")  # Light blue background

    # Title Label
    title_label = tk.Label(root, text="Real-Time Speech Recognition", font=("Helvetica", 18, "bold"), bg="#e6f7ff")
    title_label.pack(pady=20)

    # Start Prediction Button
    predict_button = tk.Button(root, text="Start Recording", font=("Helvetica", 14), bg="#007acc", fg="white",
                               command=start_prediction)
    predict_button.pack(pady=20)

    # Result Label
    result_label = tk.Label(root, text="Press 'Start Recording' to begin", font=("Helvetica", 14), bg="#e6f7ff")
    result_label.pack(pady=30)

    # Run the GUI
    root.mainloop()

# Main Execution
if __name__ == "__main__":
    MODEL_PATH = "aud_model.hdf5"  # Path to your trained model
    try:
        loaded_model = load_audio_model(MODEL_PATH)
        print("Model loaded successfully.")
        run_gui(loaded_model)
    except Exception as e:
        print(f"Error loading model: {e}")
