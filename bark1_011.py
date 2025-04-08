import customtkinter as ctk
import tkinter as tk
import threading
import sounddevice as sd
import numpy as np
import time
import math
from datetime import datetime
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import spectrogram, iirnotch, lfilter
from pydub import AudioSegment
import os

labels = [
    "Background Noise",
    "Large-dog-bark",
    "Large-dog-growl",
    "Large-dog-howl",
    "Medium-dog-bark",
    "Medium-dog-growl",
    "Medium-dog-howl",
    "Small-dog-bark",
    "Small-dog-growl",
    "Small-dog-howl"
]

model_path = "/home/bark1/soundclassifier_with_metadataV4.1.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

SAMPLERATE = 48000
CHANNELS = 1
RECORDING_DURATION = 5
MIC_DEVICE = "hw:1,0"

def notch_filter(audio, fs=48000, freq=12500, Q=30):
    # Calculate normalized frequency (w0) relative to Nyquist frequency (fs/2)
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def record_audio(duration=RECORDING_DURATION):
    audio_data = sd.rec(int(SAMPLERATE * duration),
                        samplerate=SAMPLERATE,
                        channels=CHANNELS,
                        dtype='int16',
                        device=MIC_DEVICE)
    sd.wait()
    return audio_data.flatten()

def classify_audio(audio):
    # Convert to float32 for more precise filtering
    audio_float = audio.astype(np.float32)
    # Apply the notch filter to remove 12,500 Hz interference
    filtered_audio = notch_filter(audio_float, fs=SAMPLERATE, freq=12500, Q=30)
    # Resample filtered audio to match model input length
    resampled = np.interp(np.linspace(0, len(filtered_audio), input_shape[1]),
                          np.arange(len(filtered_audio)),
                          filtered_audio)
    max_val = np.max(np.abs(resampled))
    normalized = resampled / max_val if max_val != 0 else resampled
    processed_audio = np.expand_dims(normalized.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], processed_audio)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    return labels[predicted_index], output_data

def compute_dba(audio):
    audio_float = audio.astype(np.float32)
    rms = np.sqrt(np.mean(np.square(audio_float)))
    return 20 * np.log10(rms) if rms > 0 else -np.inf

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("320x240")
        self.title("Bark Witness")
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(expand=True, fill="both")
        self.create_home_page()

    def create_home_page(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        buttons = ["Introduction", "Record Barking", "Session Summary Report", "Detailed Reports & Audio Playback", "Settings"]
        for btn_text in buttons:
            btn = ctk.CTkButton(self.content_frame, text=btn_text, command=lambda text=btn_text: self.switch_page(text), font=("Arial", 14))
            btn.pack(pady=5, padx=10, fill="x")

    def switch_page(self, page_name):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        if page_name == "Record Barking":
            self.create_record_barking_page()
        else:
            label = ctk.CTkLabel(self.content_frame, text=f"{page_name} Page", font=("Arial", 16))
            label.pack(pady=20)
            back_btn = ctk.CTkButton(self.content_frame, text="Cancel", command=self.create_home_page, font=("Arial", 12))
            back_btn.pack(pady=10)

    def create_record_barking_page(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.header_label = ctk.CTkLabel(self.content_frame, text="Preparation", font=("Arial", 16))
        self.header_label.pack(pady=2)
        self.spec_frame = ctk.CTkFrame(self.content_frame, width=280, height=60)
        self.spec_frame.pack(pady=2, fill="both", expand=False)
        self.spec_fig = Figure(figsize=(2.5, 0.8), dpi=100)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title("Live Spectrogram", fontsize=8)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=self.spec_frame)
        self.spec_canvas.draw()
        self.spec_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.elapsed_label = ctk.CTkLabel(self.content_frame, text="Elapsed: 00:00:00", font=("Arial", 12))
        self.elapsed_label.pack(pady=2)
        self.sound_label = ctk.CTkLabel(self.content_frame, text="Nuisance Sounds: 0", font=("Arial", 12))
        self.sound_label.pack(pady=2)
        self.dba_label = ctk.CTkLabel(self.content_frame, text="dBA: N/A", font=("Arial", 12))
        self.dba_label.pack(pady=2)
        self.bottom_frame = ctk.CTkFrame(self.content_frame)
        self.bottom_frame.pack(side="bottom", fill="x", pady=10)
        self.back_btn = ctk.CTkButton(self.bottom_frame, text="Back", command=self.create_home_page, font=("Arial", 12))
        self.back_btn.pack(side="left", padx=10)
        self.record_btn = ctk.CTkButton(self.bottom_frame, text="Record", command=self.start_recording, font=("Arial", 12))
        self.record_btn.pack(side="right", padx=10)
        self.is_recording = False
        self.start_time = None
        self.nuisance_count = 0
        self.audio_buffer = []
        self.rolling_buffer = []
        self.event_triggered = False
        self.last_trigger_time = 0
        self.trigger_cooldown = 10

    def start_recording(self):
        self.is_recording = True
        self.start_time = time.time()
        self.nuisance_count = 0
        self.audio_buffer = []
        self.rolling_buffer = []
        self.header_label.configure(text="Session in Progress")
        self.bottom_frame.destroy()
        self.stop_btn = ctk.CTkButton(self.content_frame, text="Stop Recording", command=self.stop_recording, font=("Arial", 14))
        self.stop_btn.pack(pady=10)
        # Increased blocksize to 1024 to reduce callback overhead on the Pi3A
        self.stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16',
                                     device=MIC_DEVICE, blocksize=1024, callback=self.audio_callback)
        self.stream.start()
        self.update_recording_info()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Stream status:", status)
        try:
            samples = indata[:, 0].tolist()
            self.audio_buffer.extend(samples)
            self.rolling_buffer.extend(samples)
            if len(self.rolling_buffer) > SAMPLERATE * 5:
                self.rolling_buffer = self.rolling_buffer[-SAMPLERATE * 5:]
        except Exception as e:
            print("Error in audio callback:", e)

    def update_recording_info(self):
        if not self.is_recording:
            return
        elapsed = time.time() - self.start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self.elapsed_label.configure(text=f"Elapsed: {h:02}:{m:02}:{s:02}")
        required_samples = input_shape[1]
        if len(self.audio_buffer) >= required_samples:
            recent_audio = np.array(self.audio_buffer[-required_samples:])
            result, output_data = classify_audio(recent_audio)
            confidence = output_data[0][np.argmax(output_data)]
            if result != "Background Noise" and confidence >= 0.5:
                self.nuisance_count += 1
                self.sound_label.configure(text=f"Nuisance Sounds: {self.nuisance_count}")
                print(f"Bark triggered: label = {result}, confidence = {confidence:.2f}")
                if not self.event_triggered and (time.time() - self.last_trigger_time > self.trigger_cooldown):
                    self.last_trigger_time = time.time()
                    self.event_triggered = True
                    pre_event = self.rolling_buffer.copy()
                    threading.Thread(target=self.handle_event_recording, args=(pre_event,), daemon=True).start()
            dba = compute_dba(recent_audio)
            self.dba_label.configure(text=f"dBA: {dba:.1f}")
            # Uncomment below to visualize the spectrogram if desired:
            """
            f_vals, t_vals, Sxx = spectrogram(recent_audio, fs=SAMPLERATE, nperseg=128, noverlap=64)
            self.spec_ax.clear()
            self.spec_ax.pcolormesh(t_vals, f_vals, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            self.spec_ax.set_ylabel('Hz', fontsize=8)
            self.spec_ax.set_xlabel('sec', fontsize=8)
            self.spec_ax.tick_params(labelsize=6)
            self.spec_fig.tight_layout()
            self.spec_canvas.draw()
            """
        self.after(1000, self.update_recording_info)

    def handle_event_recording(self, pre_event):
        # Ensure pre_event is trimmed to the last 5 seconds worth of samples
        pre_event = pre_event[-SAMPLERATE * 5:]
        time.sleep(5)
        # Explicitly take only the last 5 seconds of post-event audio
        post_event = self.rolling_buffer[-SAMPLERATE * 5:]
        combined = np.concatenate((np.array(pre_event), np.array(post_event)))
        self.save_triggered_audio(combined)
        self.event_triggered = False


    def save_triggered_audio(self, audio):
        now = datetime.now()
        folder_name = now.strftime("%Y-%m-%d_%H")
        base_folder = "Recordings"
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = now.strftime("%Y-%m-%d_%H-%M-%S") + ".mp3"
        file_path = os.path.join(folder_path, file_name)
        audio_bytes = audio.tobytes()
        segment = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=SAMPLERATE, channels=1)
        segment.export(file_path, format="mp3")

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        self.stop_btn.destroy()
        self.header_label.configure(text="Preparation")
        self.elapsed_label.configure(text="Elapsed: 00:00:00")
        self.bottom_frame = ctk.CTkFrame(self.content_frame)
        self.bottom_frame.pack(side="bottom", fill="x", pady=10)
        self.back_btn = ctk.CTkButton(self.bottom_frame, text="Back", command=self.create_home_page, font=("Arial", 12))
        self.back_btn.pack(side="left", padx=10)
        self.record_btn = ctk.CTkButton(self.bottom_frame, text="Record", command=self.start_recording, font=("Arial", 12))
        self.record_btn.pack(side="right", padx=10)

if __name__ == "__main__":
    app = App()
    app.mainloop()
