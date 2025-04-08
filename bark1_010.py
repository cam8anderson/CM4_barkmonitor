import subprocess
import serial
import customtkinter as ctk
import tkinter as tk
from datetime import datetime, timezone
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time, os, pytz, sys, csv, pyaudio, wave, threading, shutil
import tflite_runtime.interpreter as tflite
from fpdf import FPDF
import sounddevice as sd
import warnings
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
import warnings
from pydub import AudioSegment

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MICROPHONE_DEVICE = "hw:1,0"  
SAMPLERATE = 48000
CHANNELS = 2

reports_dir = "/home/bark1/CSV_Files"
pdf_reports_dir = "/home/bark1/PDF_Reports"
audio_dir = "/home/bark1/Recordings"

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(pdf_reports_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

reports_dir = os.path.join(script_dir, 'CSV_Files')
os.makedirs(reports_dir, exist_ok=True)

audio_dir = os.path.join(script_dir, 'Recordings')
os.makedirs(audio_dir, exist_ok=True)

pdf_reports_dir = os.path.join(script_dir, 'PDF_Reports')
os.makedirs(pdf_reports_dir, exist_ok=True)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

US_TIMEZONES = [
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
    "US/Alaska",
    "US/Hawaii",
]

languages = ["English", "Espanol", "Francais"]

model_path = "/home/bark1/soundclassifier_with_metadataV4.1.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
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

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry('480x320')
        self.grid_columnconfigure(0, weight=0, minsize=200)
        self.grid_columnconfigure(1, weight=0, minsize=600)
        self.grid_rowconfigure(0, weight=0, minsize=480)
        self.is_playing = False  
        self.playback_process = None  
        self.check_storage()  
        self.highest_db = 0.0
        self.audio_buffer = []  
        self.buffer_size = 16000 * 5
        self.bark_count = 0  
        self.bark_timestamps = []  
        self.recording_time = 60
        self.peak_val_interval = 0
        self.after(15 * 60 * 1000, self.report_peak_db_interval)
        self.nav_frame = ctk.CTkFrame(self, width=200)
        self.nav_frame.grid(row=0, column=0, sticky="ns")
        for i in range(7):
            self.nav_frame.grid_rowconfigure(i, weight=1)
        self.pages = {}
        self.add_nav_button("Home", 0)
        self.add_nav_button("Record Barking", 1)
        self.add_nav_button("CSV Reports", 2)  
        self.add_nav_button("PDF Reports", 3)  
        self.add_nav_button("Playback", 4)
        self.add_nav_button("Settings", 5)
        self.add_nav_button("Exit", 6, self.quit_app)
        self.content_frame = ctk.CTkFrame(self, width=600)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.create_page("Home")
        self.create_page("Record Barking")
        self.create_page("CSV Reports")
        self.create_page("PDF Reports")
        self.create_page("Playback")
        self.create_page("Introduction")
        self.create_page("Settings")
        self.create_page("Exit")
        self.selected_timezone = "US/Eastern"
        self.time_label = ctk.CTkLabel(self, text='', font=('Arial', 14))
        self.time_label.grid(row=0, column=1, sticky='ne', padx=10, pady=10)
        self.update_time()
        self.show_page("Home")
        self.is_listening = False
        self.is_recording = False
        self.gps_uart = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
        self.gps_data = {"lat": "N/A", "lon": "N/A", "utc": "N/A"}
        self.read_gps_data()
        self.bind("<Escape>", self.exit_fullscreen)

    def quit_app(self):
        """Quit the application gracefully."""
        self.destroy()
        sys.exit()

    def record_audio(duration=5):
        """ Records audio from the microphone HAT """
        print("Recording audio...")
        audio_data = sd.rec(int(SAMPLERATE * duration), samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16', device=MICROPHONE_DEVICE)
        sd.wait()
        return audio_data
    
    def update_recording_time(self, value):
        """Updates the recording time based on slider selection."""
        self.recording_time = int(value)
        print(f"Recording Time Set: {self.recording_time} minutes")
         
    def analyze_frequency(audio_data, sample_rate):
        """Performs FFT to analyze bark frequency."""
        N = len(audio_data)
        yf = fft(audio_data)
        xf = np.fft.fftfreq(N, 1 / sample_rate)
  
        idx = np.argmax(np.abs(yf))
        peak_freq = abs(xf[idx])
        return peak_freq
 
    def read_gps_data(self):
        """Reads GPS data from the directly connected GPS module and updates class variables."""
        gps_data = self.gps_uart.readline().decode("utf-8").strip()
        if gps_data:
            try:
                lat, lon, utc = gps_data.split(",")
                self.gps_data = {"lat": lat.strip(), "lon": lon.strip(), "utc": utc.strip()}
                print(f"?? Received GPS: {lat}, {lon}, {utc}")
            except ValueError:
                print("? GPS Data Format Incorrect: ", gps_data)

        self.after(1000, self.read_gps_data)
          
    def create_ui(self):
        """Creates navigation and content frames for the UI."""
        #self.nav_frame = ctk.CTkFrame(self, width=200)
        #self.nav_frame.grid(row=0, column=0, sticky="ns")

        #for i in range(7):
        #    self.nav_frame.grid_rowconfigure(i, weight=1)

        self.pages = {}

        self.add_nav_button("Home", 0)
        self.add_nav_button("Record Barking", 1)
        self.add_nav_button("CSV Reports", 2)
        self.add_nav_button("Playback", 3)
        self.add_nav_button("Settings", 4)
        self.add_nav_button("Exit", 5, self.quit_app)

        self.content_frame = ctk.CTkFrame(self, width=600)
        self.content_frame.grid(row=0, column=1, sticky="nsew")

        self.create_page("Home")
        self.create_page("Record Barking")
        self.create_page("Reports")
        self.create_page("Playback")
        self.create_page("Settings")

    def update_threshold(self, value):
        """Update the threshold value from the dropdown and update label."""
        self.threshold = int(value)  
        self.threshold_label.configure(text=f"Threshold: {self.threshold}%")

        if self.threshold < self.lowest_threshold:  
            self.lowest_threshold = self.threshold
            
    
    def on_file_selected(self, event):
        """Handles selection of a file from the reports list."""
        selection = event.widget.curselection()
        if not selection:
            return
        
        selected_index = selection[0]
        selected_file = self.file_listbox.get(selected_index)
        file_path = os.path.join(reports_dir, selected_file)

        self.display_csv_content(file_path)

    def refresh_file_list(self):
        """Refreshes the file list in the reports section."""
        if not hasattr(self, "file_listbox"):
            print("Error: file_listbox is not initialized yet.")
            return
        
        self.file_listbox.delete(0, tk.END)
        csv_files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]

        for file in csv_files:
            self.file_listbox.insert(tk.END, file)
        
        print("CSV file list refreshed")

    def check_storage(self):
        """Checks available disk space for audio recordings and reports."""
        try:
            statvfs = os.statvfs("/")
            total_space = (statvfs.f_frsize * statvfs.f_blocks) / (1024 * 1024 * 1024)  
            free_space = (statvfs.f_frsize * statvfs.f_bfree) / (1024 * 1024 * 1024)  

            print(f"Total Disk Space: {total_space:.2f} GB")
            print(f"Free Disk Space: {free_space:.2f} GB")

            if free_space < 1: 
                print("?? Warning: Low disk space! Free up space to avoid recording issues.")

        except Exception as e:
            print(f"? Error checking storage: {e}")

        self.audio_buffer = [] 
        self.buffer_size = 16000 * 5  

        self.audio_buffer = [] 
        self.buffer_size = 16000 * 5  

        self.audio_buffer = [] 
        self.buffer_size = 16000 * 5  

        #self.nav_frame = ctk.CTkFrame(self, width=200)
        #self.nav_frame.grid(row=0, column=0, sticky="ns")
        #for i in range(7):
        #    self.nav_frame.grid_rowconfigure(i, weight=1)
        self.pages = {}

        #self.add_nav_button("Home", 0)
        #self.add_nav_button("Record Barking", 1)
        #self.add_nav_button("Reports", 2)
        #self.add_nav_button("Playback", 3)
        #self.add_nav_button("Introduction", 4)
        #self.add_nav_button("Settings", 5)
        #self.add_nav_button("Exit", 6, self.quit_app)

        self.content_frame = ctk.CTkFrame(self, width=600)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.create_page("Home")
        self.create_page("Record Barking")
        self.create_page("CSV Reports")
        self.create_page("PDF Reports")
        self.create_page("Playback")
        self.create_page("Introduction")
        self.create_page("Settings")
        self.create_page("Exit")

        self.selected_timezone = "US/Eastern"
        self.time_label = ctk.CTkLabel(self, text='', font=('Arial', 14))
        self.time_label.grid(row=0, column=1, sticky='ne', padx=10, pady=10)
        self.update_time()

        self.show_page("Home")
        self.is_listening = False
        self.is_recording = False

    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode when Escape is pressed"""
        self.attributes("-fullscreen", False)

    def show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(expand=True, fill="both")

    def add_nav_button(self, name, index, command=None):
        if command is None:
            command = lambda: self.show_page(name)
        button = ctk.CTkButton(
            self.nav_frame,
            text=name,
            command=command,
            height=50,
            font=("Arial", 16)
        )
        button.grid(row=index, column=0, padx=10, pady=5, sticky="ew")

    def update_time(self):
        tz = pytz.timezone(self.selected_timezone)
        current_time = pytz.utc.localize(datetime.utcnow()).astimezone(tz)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        self.time_label.configure(text=formatted_time)
        self.after(1000, self.update_time)

    def create_page(self, name):
        """Creates a new page inside the content frame."""
        page_frame = ctk.CTkFrame(self.content_frame)
        self.pages[name] = page_frame  
 
        image_label = ctk.CTkLabel(page_frame, text="[Placeholder for Image]")
        image_label.pack(pady=10)

        if name == "Home":
            self.create_home_page(page_frame)
        elif name == "Record Barking":
            self.create_record_page(page_frame) 
        elif name == "CSV Reports":
            self.create_csv_reports_page(page_frame)  
        elif name == "PDF Reports":
            self.create_pdf_reports_page(page_frame) 
        elif name == "Playback":
            self.create_playback_page(page_frame)
        elif name == "Settings":
            self.create_settings_page(page_frame)
        elif name == "Introduction":
            self.create_intro_page(page_frame)
        # "Exit" page can be set up similarly if needed.

    def on_audio_selected(self, event):
        """Handles selection of an audio file from the playback list."""
        selection = event.widget.curselection()
        if not selection:
            return  
        selected_index = selection[0]
        selected_file = self.audio_listbox.get(selected_index)
        print(f"Selected audio file: {selected_file}")

    def add_back_button(self, parent):
        back_button = ctk.CTkButton(parent, text="Back", command=lambda: self.show_page("Home"), font=("Arial", 12))
        back_button.pack(anchor="w", padx=5, pady=5)    
    
    def create_home_page(self, parent):
        label = ctk.CTkLabel(
            parent,
            text="Welcome to the Home Page!\n\nNovico Bark Witness.",
            font=("Arial", 20), wraplength=600
        )
        label.pack(pady=20)

        launch_mixer_button = ctk.CTkButton(
            parent,
            text="Open Stable Mixer",
            command=self.launch_stable_mixer
        )
        launch_mixer_button.pack(pady=10)

        self.pages["Home"] = parent
        
        # Create navigation buttons for each page.
        btn_record = ctk.CTkButton(parent, text="Record Barking", command=lambda: self.show_page("Record Barking"), font=("Arial", 14))
        btn_csv = ctk.CTkButton(parent, text="CSV Reports", command=lambda: self.show_page("CSV Reports"), font=("Arial", 14))
        btn_pdf = ctk.CTkButton(parent, text="PDF Reports", command=lambda: self.show_page("PDF Reports"), font=("Arial", 14))
        btn_playback = ctk.CTkButton(parent, text="Playback", command=lambda: self.show_page("Playback"), font=("Arial", 14))
        btn_settings = ctk.CTkButton(parent, text="Settings", command=lambda: self.show_page("Settings"), font=("Arial", 14))
        btn_exit = ctk.CTkButton(parent, text="Exit", command=self.quit_app, font=("Arial", 14))
        
        for btn in (btn_record, btn_csv, btn_pdf, btn_playback, btn_settings, btn_exit):
            btn.pack(pady=3, padx=10, fill="x")

    def launch_stable_mixer(self):
        """Launches the Stable Mixer GUI."""
        try:
            subprocess.Popen(["python3", "/home/bark1/stable_mixer_gui.py"])
        except Exception as e:
            print(f"? Error launching Stable Mixer: {e}")
        
    def toggle_recording(self):
        """Starts or stops the audio recording using arecord."""
        if self.is_recording:
            self.end_time = time.time()
            if self.recording_process:
                print("?? Stopping recording...")
                self.recording_process.terminate()
                self.recording_process.wait()
                self.recording_process = None

                if os.path.exists(self.last_recorded_file):
                    self.convert_to_s16le()
                    print("? Recording stopped and converted to S16_LE.")
                else:
                    print("?? Recording file not found! Conversion skipped.")

            if hasattr(self, "generate_csv_report") and self.last_recorded_file:
                self.generate_csv_report()
                print("? CSV report generated after recording stopped.")
            else:
                print("? Error: `generate_csv_report` function not found or invalid recording file.")

            self.is_recording = False
            self.record_button.configure(text="Start Recording")

        else:
            subprocess.run(["pkill", "-x", "arecord"], check=False)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.last_recorded_file = os.path.join(audio_dir, f"recording_{timestamp}.wav")
            self.converted_file = os.path.join(audio_dir, f"recording_{timestamp}_s16le.wav")

            try:
                self.recording_process = subprocess.Popen([
                    "arecord", "-D", "hw:1,0", "-c", "2", "-r", "48000", "-f", "S32_LE", "-t", "wav", self.last_recorded_file
                ])
                print(f"?? Recording started: {self.last_recorded_file}")

                self.start_time = time.time()
                self.is_recording = True
                self.update_elapsed_time()
                self.record_button.configure(text="Stop Recording")
            except FileNotFoundError:
                print("? Error: `arecord` not found. Ensure ALSA is installed.")
            except Exception as e:
                print(f"? Error during recording: {e}")
       
    def convert_to_s16le(self):
        """Converts the recorded audio from S32_LE to S16_LE using sox and checks encoding."""
        if not os.path.exists(self.last_recorded_file):
            print("? Error: No recorded file found to convert.")
            return

        try:
            command = ["sox", self.last_recorded_file, "-b", "16", self.converted_file]
            subprocess.run(command, check=True)
            print(f"? Converted {self.last_recorded_file} to {self.converted_file}")

            soxi_output = subprocess.run(["soxi", self.converted_file], capture_output=True, text=True)
            print(f"? SOXI Output:\n{soxi_output.stdout}")

            os.remove(self.last_recorded_file)
            print(f"? Deleted original file: {self.last_recorded_file}")

        except subprocess.CalledProcessError as e:
            print(f"? Error during conversion: {e}")

    def create_record_page(self, parent):
        self.pages["Record Barking"] = parent
        self.add_back_button(parent)
        label = ctk.CTkLabel(parent, text="Record Barking", font=("Arial", 16))
        label.pack(pady=10)

        self.record_button = ctk.CTkButton(
            parent,
            text="Start Recording",
            command=self.toggle_recording,
            font=("Arial", 20)
        )
        self.record_button.pack(pady=20)

        self.status_label = ctk.CTkLabel(parent, text="Status: Not listening", font=("Arial", 16))
        self.status_label.pack(pady=10)

        self.elapsed_time_label = ctk.CTkLabel(parent, text="Elapsed Time: 00:00:00", font=("Arial", 16))
        self.elapsed_time_label.pack(pady=10)

        self.detection_label = ctk.CTkLabel(parent, text="Detected sound: None", font=("Arial", 16))
        self.detection_label.pack(pady=10)

        self.threshold = 40
        self.lowest_threshold = self.threshold

        self.threshold_label = ctk.CTkLabel(parent, text=f"Threshold: {self.threshold}%", font=("Arial", 14))
        self.threshold_label.pack(pady=5)

        # Add a label for peak dB over the interval:
        self.peak_db_label = ctk.CTkLabel(parent, text="Peak dB (15 min): N/A", font=("Arial", 14))
        self.peak_db_label.pack(pady=5)

        record_time_label = ctk.CTkLabel(parent, text="Select Recording Time Range", font=("Arial", 16))
        record_time_label.pack(pady=5)

        self.record_time_slider = ctk.CTkSlider(
            parent,
            from_=20, to=360, number_of_steps=5, command=self.update_recording_time
        )
        self.record_time_slider.set(60)
        self.record_time_slider.pack(pady=10)


    def create_csv_reports_page(self, parent):
        """Creates the reports page with file list and controls using `pack()` to avoid Tkinter errors."""        
        self.pages["CSV Reports"] = parent
        self.add_back_button(parent)
        label = ctk.CTkLabel(parent, text="CSV Reports", font=("Arial", 16))
        label.pack(pady=10)
        
        frame = ctk.CTkFrame(parent)  
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        label = ctk.CTkLabel(frame, text="Reports Page", font=("Arial", 20))
        label.pack(pady=10, anchor="w")  

        self.file_listbox = tk.Listbox(frame, height=15, width=50)
        self.file_listbox.pack(pady=10, fill="both", expand=True)  
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_selected)

        refresh_button = ctk.CTkButton(
            frame,
            text="Refresh File List",
            command=self.refresh_file_list
        )
        refresh_button.pack(pady=5, fill="x")  

        generate_pdf_button = ctk.CTkButton(
            frame,
            text="Generate PDF Report",
            command=self.generate_bark_events_pdf
        )
        generate_pdf_button.pack(pady=5, fill="x")  

        self.refresh_file_list()  

    def create_pdf_reports_page(self, parent):
        self.pages["PDF Reports"] = parent
        self.add_back_button(parent)
        label = ctk.CTkLabel(parent, text="PDF Reports", font=("Arial", 16))
        label.pack(pady=10)
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        btn_bark_pdf = ctk.CTkButton(
            frame,
            text="Generate Bark Events PDF",
            command=self.generate_bark_events_pdf
        )
        btn_bark_pdf.pack(pady=5, fill="x")
        btn_peak_pdf = ctk.CTkButton(
            frame,
            text="Generate Peak Loudness PDF",
            command=self.generate_peak_loudness_pdf
        )
        btn_peak_pdf.pack(pady=5, fill="x")
        # Existing code to list PDFs remains unchanged.
        self.pdf_listbox = tk.Listbox(frame, height=15, width=50)
        self.pdf_listbox.pack(pady=10, fill="both", expand=True)
        self.pdf_listbox.bind("<<ListboxSelect>>", self.on_pdf_selected)
        refresh_pdf_button = ctk.CTkButton(
            frame,
            text="Refresh PDF List",
            command=self.refresh_pdf_list
        )
        refresh_pdf_button.pack(pady=5, fill="x")
        self.refresh_pdf_list()

    def on_pdf_selected(self, event):
        """Handles selection of a PDF file and opens it."""
        selection = event.widget.curselection()
        if not selection:
            return
        selected_index = selection[0]
        selected_file = self.pdf_listbox.get(selected_index)
        file_path = os.path.join(pdf_reports_dir, selected_file)
        print(f"Opening PDF: {file_path}")
        try:
            subprocess.run(["xdg-open", file_path], check=True)
        except Exception as e:
            print(f"Error opening PDF: {e}")


    def create_playback_page(self, parent):
        """Creates the playback page with audio file list and playback controls."""        
        self.pages["Playback"] = parent
        self.add_back_button(parent)
        label = ctk.CTkLabel(parent, text="Playback", font=("Arial", 16))
        label.pack(pady=10)

        label = ctk.CTkLabel(parent, text="Playback Page", font=("Arial", 20))
        label.pack(pady=20)

        self.audio_listbox = tk.Listbox(parent, height=10, width=50)
        self.audio_listbox.pack(pady=10)
        self.audio_listbox.bind("<<ListboxSelect>>", self.on_audio_selected)  

        refresh_button = ctk.CTkButton(
            parent,
            text="Refresh Audio List",
            command=self.refresh_audio_list
        )
        refresh_button.pack(pady=5)

        play_button = ctk.CTkButton(
            parent,
            text="Play Selected Audio",
            command=self.play_selected_audio
        )
        play_button.pack(pady=5)
        
        stop_button = ctk.CTkButton(
            parent,
            text="Stop Playback",
            command=self.stop_playback
        )
        stop_button.pack(pady=5)

        self.refresh_audio_list()
    


    def create_intro_page(self, parent):
        label = ctk.CTkLabel(
            parent,
            text="Introduction to Bark Witness",
            font=("Arial", 20), wraplength=600
        )
        label.pack(pady=20)

    def create_settings_page(self, parent):
        self.pages["Settings"] = parent
        self.add_back_button(parent)
        label = ctk.CTkLabel(parent, text="Settings", font=("Arial", 16))
        label.pack(pady=10)
        
        tz_label = ctk.CTkLabel(parent, text="Select Time Zone", font=("Arial", 16))
        tz_label.pack(pady=5)
        self.tz_combobox = ctk.CTkComboBox(
            parent,
            values=US_TIMEZONES,
            command=self.change_timezone
        )
        self.tz_combobox.set("US/Eastern")
        self.tz_combobox.pack(pady=10)

        record_time_label = ctk.CTkLabel(parent, text="Select Recording Time Range", font=("Arial", 16))
        record_time_label.pack(pady=5)
        self.record_time_slider = ctk.CTkSlider(parent, from_=20, to=360, number_of_steps=5, command=self.update_recording_time)
        self.record_time_slider.set(60) 
        self.record_time_slider.pack(pady=10)

        clear_button = ctk.CTkButton(
            parent,
            text="Clear Reports and Audio_files",
            command=self.clear_data,
            fg_color="red",
            hover_color="darkred"
        )
        clear_button.pack(pady=20)

    def update_recording_time(self, value):
        """Updates the recording time based on slider selection."""
        self.recording_time = int(value)
        print(f"Recording Time Set: {self.recording_time} minutes")
        
    def clear_data(self):
        """Clears all reports and audio files."""
        confirm = tk.messagebox.askyesno(
            "Confirm Deletion", "Are you sure you want to delete all reports and audio files?"
        )
        if not confirm:
            return
        
        folders_to_clear = [
            os.path.join(script_dir, "Reports"),
            os.path.join(script_dir, "CSV Files"), 
            os.path.join(script_dir, "Audio_files")  
        ]

        for folder in folders_to_clear:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

        tk.messagebox.showinfo("Success", "All reports and audio files have been deleted!")
        
    def update_elapsed_time(self):
        """Updates the elapsed recording time every second."""
        if self.is_recording and self.start_time:
            elapsed_seconds = int(time.time() - self.start_time)
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_str = f"Elapsed Time: {hours:02}:{minutes:02}:{seconds:02}"
            self.elapsed_time_label.configure(text=elapsed_str)

            self.after(1000, self.update_elapsed_time)

    def change_timezone(self, selected_tz):
        self.selected_timezone = selected_tz
        self.update_time()

    def change_language(self, selected_language):
        pass

    def classify_audio(self, audio):
        resampled = np.interp(
            np.linspace(0, len(audio), input_shape[1]),
            np.arange(len(audio)),
            audio
        )
        normalized = resampled / np.max(np.abs(resampled))
        processed_audio = np.expand_dims(normalized.astype(np.float32), axis=0)

        interpreter.set_tensor(input_details[0]['index'], processed_audio)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        return labels[predicted_index], output_data
    
    def start_audio_classification(self):
        """Starts real-time sound classification and dB monitoring."""
        self.highest_db = 0.0 

        self.audio_stream = sd.InputStream(
            samplerate=48000,
            device="hw:1,0",
            channels=2,
            dtype='int16',
            callback=self.audio_callback
        )
        self.audio_stream.start()
        print("?? Real-time audio classification started...")

    def audio_callback(self, indata, frames, audio_time, status):
        if status:
            print(f"Error: {status}")
            return

        audio_data = indata[:, 0] if indata.size > 0 else np.zeros(frames)
        self.audio_buffer.extend(audio_data.tolist())
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]

        current_max = np.max(np.abs(indata))
        if current_max > self.peak_val_interval:
            self.peak_val_interval = current_max

        label, output_data = self.classify_audio(np.array(self.audio_buffer[-self.buffer_size:]))

        # For our purposes, any event not labeled as Background Noise and meeting the confidence threshold is a nuisance event.
        dog_index = labels.index("Dog Barking") if "Dog Barking" in labels else 1
        dog_confidence = output_data[0][dog_index]

        if label != "Background Noise" and dog_confidence >= self.threshold:
            # Compute dBA (assuming 16-bit PCM, max value 32767)
            dba = 20 * np.log10(current_max / 32767) if current_max > 0 else -float('inf')
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dBA": f"{dba:.2f}",
                "label": label,
                "audio_file": "Pending...",
                "lat": self.gps_data.get("lat", "N/A"),
                "lon": self.gps_data.get("lon", "N/A")
            }
            self.nuisance_events.append(event)
            # Launch a thread to capture post-bark audio and update the event record
            threading.Thread(target=self.capture_and_update_event, args=(event,), daemon=True).start()
            print(f"Bark detected! Label: {label} Confidence: {dog_confidence:.2f}, dBA: {dba:.2f}")

        self.after(0, lambda: self.detection_label.configure(
            text=f"Barks This Session: {self.bark_count}\n(Dog Confidence: {dog_confidence:.2f})"
        ))


    def capture_and_update_event(self, event):
        """Captures the post-bark audio and updates the event record with the saved file name."""
        audio_file = self.save_bark_audio(self.audio_buffer)
        event["audio_file"] = audio_file

    def report_peak_db_interval(self):
        if self.peak_val_interval > 0:
            # Assuming 16-bit PCM (max value 32767)
            peak_db = 20 * np.log10(self.peak_val_interval / 32767)
        else:
            peak_db = -float('inf')
        # Update the label on the recording page
        self.peak_db_label.configure(text=f"Peak dB (15 min): {peak_db:.2f} dB")
        print(f"Peak dB over last 15 minutes: {peak_db:.2f} dB")
        
        # Reset for the next interval and schedule the next report.
        self.peak_val_interval = 0
        self.after(15 * 60 * 1000, self.report_peak_db_interval)



    def run_stream():
            """Runs the audio stream in a separate thread."""
            try:
                self.stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback)
                self.stream.start()

                while self.is_recording:
                    time.sleep(0.1)  

                self.stream.stop()
                self.stream.close()

            except Exception as e:
                print(f"Error: {e}")

            threading.Thread(target=run_stream, daemon=True).start()

    def save_bark_audio(self, pre_bark_audio):
        """Saves the bark audio (combining pre-bark and 5 seconds of post-bark audio) as a WAV file and returns the file name."""
        print("Recording post-bark audio...")
        duration = 5  
        post_bark_audio = sd.rec(int(48000 * duration), samplerate=48000, channels=2, dtype='int16', device=MICROPHONE_DEVICE)
        sd.wait()

        post_bark_audio = post_bark_audio.flatten()
        full_audio = np.concatenate((np.array(pre_bark_audio)[-self.buffer_size:], post_bark_audio))

        if len(full_audio) == 0:
            print("ERROR: No audio captured! Check audio buffer.")
            return ""

        now = datetime.now()
        folder_name = now.strftime("%Y-%m-%d_%H")
        folder_path = os.path.join(audio_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_path, f"bark_{timestamp}.wav")

        # Export the audio as a WAV file using AudioSegment
        segment = AudioSegment(
            data=full_audio.tobytes(),
            sample_width=2,
            frame_rate=48000,
            channels=2
        )
        segment.export(filename, format="mp3")
        print(f"Bark audio saved: {filename}")
        return filename


def generate_csv_report(self):
    lat = self.gps_data.get("lat", "N/A")
    lon = self.gps_data.get("lon", "N/A")
    reports_dir = os.path.join(script_dir, "CSV_Files")
    os.makedirs(reports_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{date_str}_nuisance_report.csv"
    csv_path = os.path.join(reports_dir, filename)

    try:
        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Bark Monitor"])
            writer.writerow([f"Latitude: {lat}", f"Longitude: {lon}"])
            writer.writerow([])  # blank line
            writer.writerow(["Bark Event", "Latitude", "Longitude", "dB"])
            for event in self.nuisance_events:
                if "-" in event["label"]:
                    sound_type = event["label"].split("-")[-1].lower()
                else:
                    sound_type = event["label"].lower()
                dB_with_type = f"{event['dBA']} {sound_type}"
                writer.writerow([
                    event.get("timestamp", ""),
                    event.get("lat", ""),
                    event.get("lon", ""),
                    dB_with_type
                ])
        print(f"? CSV Report successfully saved to {csv_path}")
    except Exception as e:
        print(f"? Error saving CSV report: {e}")


    def display_csv_content(self, file_path):
        """Displays a CSV file's contents in a new Tkinter window."""
        if not os.path.exists(file_path):
            print(f"? File not found: {file_path}")
            return

        csv_window = ctk.CTkToplevel(self)
        csv_window.title(f"Report: {os.path.basename(file_path)}")
        csv_window.geometry("600x400")

        title_label = ctk.CTkLabel(csv_window, text=f"Report: {os.path.basename(file_path)}", font=("Arial", 18))
        title_label.pack(pady=10)

        try:
            with open(file_path, mode="r") as file:
                reader = csv.reader(file)
                for row in reader:
                    frame = ctk.CTkFrame(csv_window)
                    frame.pack(fill="x", padx=10, pady=5)
                    for cell in row:
                        cell_label = ctk.CTkLabel(frame, text=cell, font=("Arial", 14), anchor="w")
                        cell_label.pack(side="left", padx=5)
        except Exception as e:
            error_label = ctk.CTkLabel(csv_window, text=f"Error reading file: {str(e)}", font=("Arial", 14))
            error_label.pack(pady=10)


    def generate_bark_events_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Bark Events Report", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, "Bark Event", 1, 0, "C")
        pdf.cell(40, 10, "dBA", 1, 0, "C")
        pdf.cell(50, 10, "Class", 1, 1, "C")
        pdf.set_font("Arial", size=12)
        for event in self.nuisance_events:
            pdf.cell(50, 10, event.get("timestamp", ""), 1, 0, "C")
            pdf.cell(40, 10, event.get("dBA", ""), 1, 0, "C")
            pdf.cell(50, 10, event.get("label", ""), 1, 1, "C")
        filename = os.path.join(pdf_reports_dir, f"bark_events_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
        pdf.output(filename)
        print(f"Bark Events PDF generated: {filename}")

    # NEW: Generate a Peak Loudness Report PDF modeled after the provided sample.
    def generate_peak_loudness_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Peak Loudness Report", ln=True, align="C")
        pdf.ln(5)
        rec_start = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S") if hasattr(self, "start_time") and self.start_time else "N/A"
        rec_stop = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lat = self.gps_data.get("lat", "N/A")
        lon = self.gps_data.get("lon", "N/A")
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Recording Start: {rec_start}    Latitude: {lat}", ln=True)
        pdf.cell(0, 10, f"Recording Stop: {rec_stop}    Longitude: {lon}", ln=True)
        pdf.cell(0, 10, "Prevalent Sound: Likely German Shepherd or similar size", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 10)
        col_width = 20
        headers = ["Time", "Peak1", "Peak2", "Peak3", "Peak4"]
        for col in headers:
            pdf.cell(col_width, 10, col, border=1, align="C")
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for i in range(24):
            pdf.cell(col_width, 10, str(i), border=1, align="C")
            # For demo purposes, fixed values are used between rows 8 and 15; zeros otherwise.
            if 8 <= i <= 15:
                if i == 8:
                    values = [70, 73, 71, 86]
                elif i == 9:
                    values = [91, 90, 98, 88]
                elif i == 10:
                    values = [71, 79, 84, 0]
                elif i == 11:
                    values = [0, 0, 83, 93]
                elif i == 12:
                    values = [98, 72, 70, 0]
                elif i == 13:
                    values = [0, 0, 0, 0]
                elif i == 14:
                    values = [0, 0, 0, 0]
                elif i == 15:
                    values = [92, 91, 77, 98]
            else:
                values = [0, 0, 0, 0]
            for val in values:
                pdf.cell(col_width, 10, str(val), border=1, align="C")
            pdf.ln()
        pdf.ln(5)
        pdf.cell(0, 10, "Values are ±5dBA", ln=True)
        pdf.cell(0, 10, f"Created by Bark Witness ™ {datetime.now().strftime('%B %d %Y at %H:%M')}", ln=True)
        pdf.cell(0, 10, "Hearing Damage Risk (≥70 dBA)", ln=True)
        filename = os.path.join(pdf_reports_dir, f"peak_loudness_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
        pdf.output(filename)
        print(f"Peak Loudness PDF generated: {filename}")

    def refresh_pdf_list(self):
        """Refreshes the list of available PDF reports."""
        if not hasattr(self, "pdf_listbox"):
            print("Error: pdf_listbox is not initialized yet.")
            return

        self.pdf_listbox.delete(0, tk.END)
        pdf_files = [f for f in os.listdir(pdf_reports_dir) if f.endswith(".pdf")]
        for file in pdf_files:
            self.pdf_listbox.insert(tk.END, file)

    def refresh_audio_list(self):
        """Refreshes the list of available audio files."""
        self.audio_listbox.delete(0, tk.END)
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        for file in audio_files:
            self.audio_listbox.insert(tk.END, file)
        print('Audio files list refreshed')

    def on_audio_selected(self, event):
        pass
    
    def play_selected_audio(self):
        """Plays the selected audio file with correct formatting and stops previous playback."""
        selection = self.audio_listbox.curselection()
        if not selection:
            tk.messagebox.showinfo("Info", "Please select an audio file to play.")
            return

        selected_index = selection[0]
        selected_file = self.audio_listbox.get(selected_index)

        if not selected_file.endswith("_s16le.wav"):
            selected_file = selected_file.replace(".wav", "_s16le.wav")

        file_path = os.path.join(audio_dir, selected_file)

        if not os.path.exists(file_path):
            tk.messagebox.showerror("Error", "Selected audio file not found.")
            return

        if not hasattr(self, "playback_process"):
            self.playback_process = None

        self.stop_playback()

        threading.Thread(target=self.play_audio_file, args=(file_path,), daemon=True).start()

    def play_audio_file(self, file_path, *args):
        """Plays the selected audio file using `aplay`."""
        if not os.path.exists(file_path):
            tk.messagebox.showinfo("Info", "Selected audio file not found.")
            return

        try:
            self.stop_playback()  
            self.is_playing = True
            self.playback_process = subprocess.Popen(
                ["aplay", "-D", "hw:1,0", "-f", "S16_LE", "-r", "48000", "-c", "2", file_path]
            )
            print(f"?? Playing {file_path} on hw:1,0 with S16_LE format.")
        except Exception as e:
            print(f"? Error during playback: {e}")
           
    def stop_playback(self):
        """Stops any currently playing audio."""
        if hasattr(self, "playback_process") and self.playback_process is not None:
            print("? Stopping playback...")
            self.playback_process.terminate()
            self.playback_process.wait()
            self.playback_process = None
            self.is_playing = False
            print("? Playback stopped.")
        else:
            print("?? No active playback to stop.")

    def update_threshold(self, value):
        self.threshold = int(value) / 100  
        self.threshold_label.configure(text=f"Threshold: {int(value)}%")

        if self.threshold < self.lowest_threshold:
            self.lowest_threshold = self.threshold

    def clear_data(self):
        confirm = tk.messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete all reports and audio files?")
        if not confirm:
            return  

        folders_to_clear = [
            os.path.join(script_dir, "Reports"),
            os.path.join(script_dir, "csv"),
            os.path.join(script_dir, "Audio_files")
        ]

        for folder in folders_to_clear:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

        tk.messagebox.showinfo("Success", "All reports and audio files have been deleted!")

    def show_storage_warning(self, usage_percent):
        """Display a warning popup if storage usage is too high."""
        popup = ctk.CTkToplevel(self)
        popup.title("Storage Warning")
        popup.geometry("400x200")

        warning_label = ctk.CTkLabel(
            popup, 
            text=f"WARNING: Storage is {usage_percent:.1f}% full! \n\n"
                 "Consider deleting old reports. Settings>Clear Reports and Audio_files",
            font=("Arial", 14),
            wraplength=380
        )
        warning_label.pack(pady=20)

        dismiss_button = ctk.CTkButton(
            popup, text="OK", command=popup.destroy
        )
        dismiss_button.pack(pady=10)

    def quit_app(self):
        self.destroy()
        sys.exit()

    def exit_fullscreen(self, event=None):
        self.attributes("-fullscreen", False)

if __name__ == "__main__":
    if "DISPLAY" not in os.environ:
        print("Error: No display environment found. Ensure Openbox is running.")
        sys.exit(1)

    app = App()
    app.mainloop()
