import customtkinter
import tkinter as tk
import pyaudio
import numpy as np
from tflite_runtime.interpreter import Interpreter
from collections import deque
import threading

classes = {
    0: "Australian-shepherd",
    1: "Background Noise",
    2: "Beagle",
    3: "Boxer",
    4: "Bulldog",
    5: "Cane- corso",
    6: "Cavalier-King-Charles-spaniel",
    7: "Daschund",
    8: "Doberman-pinscher",
    9: "French-bulldog",
    10: "German-Shepherd",
    11: "German-shorthaired-pointer",
    12: "Golden-retriever",
    13: "Great-dane",
    14: "Miniature-schnauzer",
    15: "People",
    16: "Poodle",
    17: "Rottweiler",
    18: "Shih-tzu",
    19: "Welsh-corgi",
    20: "Yorkshire-terrier"
}

allowed_classes = classes

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Classification Test")
        self.geometry("800x480")
        self.is_running = False
        self.start_stop_button = customtkinter.CTkButton(self, text="Start Classification", command=self.toggle_classification)
        self.start_stop_button.pack(pady=10)
        self.labels_frame = customtkinter.CTkFrame(self)
        self.labels_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.class_labels = {}
        allowed_keys = list(allowed_classes.keys())
        n = len(allowed_keys)
        columns = 2
        rows = (n + columns - 1) // columns
        for idx, i in enumerate(allowed_keys):
            row = idx % rows
            col = idx // rows
            label = customtkinter.CTkLabel(self.labels_frame, text=f"{allowed_classes[i]}: 0.0% (max: 0.0%)", font=("Helvetica", 16))
            label.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            self.class_labels[i] = label
        self.highest_confidences = {i: 0.0 for i in allowed_classes}
        self.interpreter = Interpreter(model_path="soundclassifier_with_metadataV3.1.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.expected_length = int(self.input_shape[1])
        self.CHUNK = 1024
        self.RATE = 16000
        self.audio_buffer = deque(maxlen=2066 * 3)
        self.buffer_lock = threading.Lock()
        self.audio_thread_obj = None

    def toggle_classification(self):
        if not self.is_running:
            self.is_running = True
            self.start_stop_button.configure(text="Stop Classification")
            self.audio_thread_obj = threading.Thread(target=self.audio_thread, daemon=True)
            self.audio_thread_obj.start()
            self.update_loop()
        else:
            self.is_running = False
            self.start_stop_button.configure(text="Start Classification")
            if self.audio_thread_obj is not None:
                self.audio_thread_obj.join()
                self.audio_thread_obj = None

    def audio_thread(self):
        p = pyaudio.PyAudio()
        input_device_index = 1
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, input_device_index=input_device_index, frames_per_buffer=self.CHUNK)
        while self.is_running:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                with self.buffer_lock:
                    self.audio_buffer.extend(samples)
            except Exception as e:
                print("Error in audio thread:", e)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def update_loop(self):
        if not self.is_running:
            return
        try:
            with self.buffer_lock:
                if len(self.audio_buffer) < self.expected_length:
                    current_audio = None
                else:
                    current_audio = list(self.audio_buffer)[-self.expected_length:]
            if current_audio is not None:
                audio = np.array(current_audio, dtype=np.float32)
                audio = np.expand_dims(audio, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], audio)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                for i in allowed_classes.keys():
                    confidence = output_data[i]
                    if confidence > self.highest_confidences[i]:
                        self.highest_confidences[i] = confidence
                    current_pct = confidence * 100
                    max_pct = self.highest_confidences[i] * 100
                    self.class_labels[i].configure(text=f"{allowed_classes[i]}: {current_pct:.1f}% (max: {max_pct:.1f}%)")
        except Exception as e:
            print("Error during update_loop:", e)
        self.after(100, self.update_loop)

if __name__ == "__main__":
    app = App()
    app.mainloop()
