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

model_path = "/home/bark1/soundclassifier_with_metadataV2.1.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
labels = ["Background Noise", "Dog Barking", "People Talking"]

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry('800x480')
        self.grid_columnconfigure(0, weight=0, minsize=200)
        self.grid_columnconfigure(1, weight=0, minsize=600)
        self.grid_rowconfigure(0, weight=0, minsize=480)
        self.is_playing = False  
        self.playback_process = None  
        self.check_storage()  
        self.highest_db = 0.0
        self.audio_buffer = []  
        self.buffer_size = 16000 * 3
        self.bark_count = 0  
        self.bark_timestamps = []  
        self.recording_time = 60
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
        self.nav_frame = ctk.CTkFrame(self, width=200)
        self.nav_frame.grid(row=0, column=0, sticky="ns")

        for i in range(7):
            self.nav_frame.grid_rowconfigure(i, weight=1)

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
            
    def quit_app(self):
        """Quit the application gracefully."""
        self.destroy()
        sys.exit()
    
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
        self.buffer_size = 16000 * 3  

        self.audio_buffer = [] 
        self.buffer_size = 16000 * 3  

        self.audio_buffer = [] 
        self.buffer_size = 16000 * 3  

        self.nav_frame = ctk.CTkFrame(self, width=200)
        self.nav_frame.grid(row=0, column=0, sticky="ns")
        for i in range(7):
            self.nav_frame.grid_rowconfigure(i, weight=1)
        self.pages = {}

        self.add_nav_button("Home", 0)
        self.add_nav_button("Record Barking", 1)
        self.add_nav_button("Reports", 2)
        self.add_nav_button("Playback", 3)
        self.add_nav_button("Introduction", 4)
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

    def create_record_page(self, parent):
        """Creates the Record Barking page with recording controls."""
        label = ctk.CTkLabel(parent, text="Record Barking", font=("Arial", 20))
        label.pack(pady=20)
 
        self.record_button = ctk.CTkButton(
            parent,
            text="Start Recording",
            command=self.toggle_recording,
            font=("Arial", 20)
        )
        self.record_button.pack(pady=20)

    def on_audio_selected(self, event):
        """Handles selection of an audio file from the playback list."""
        selection = event.widget.curselection()
        if not selection:
            return  

        selected_index = selection[0]
        selected_file = self.audio_listbox.get(selected_index)
        print(f"Selected audio file: {selected_file}")

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
        """Creates the Record Barking page with recording controls and recording time slider."""
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

        self.threshold_dropdown = ctk.CTkComboBox(
            parent,
            values=[str(i) for i in range(100, -1, -5)], 
            command=self.update_threshold
        )
        self.threshold_dropdown.set(str(self.threshold))  
        self.threshold_dropdown.pack(pady=5)

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
            command=self.generate_pdf_report
        )
        generate_pdf_button.pack(pady=5, fill="x")  

        self.refresh_file_list()  
        
    def generate_csv_report(self):
         # Ensure the 'reports' directory exists
        reports_dir = os.path.join(script_dir, "csv")
        os.makedirs(reports_dir, exist_ok=True)
   
        # Filename based on current date
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{date_str}_report.csv"
        csv_path = os.path.join(reports_dir, filename)
   
        # Session Summary Data
        start_time_str = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
        session_end_str = datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S")
   
        elapsed_time = time.time() - self.start_time
        hours_elapsed = elapsed_time / 3600
        avg_barks_per_hour = self.bark_count / hours_elapsed if hours_elapsed > 0 else 0
   
        # Calculate peak 15-minute incident (placeholder calculation)
        if self.bark_timestamps:
            intervals = [t // 900 for t in self.bark_timestamps]  # Group into 15-min intervals
            peak_interval = max(set(intervals), key=intervals.count)
            peak_15_min_incident_time = datetime.fromtimestamp(
                self.start_time + peak_interval * 900
            ).strftime("%d %B %Y from %H:%M:%S")
            peak_15_min_barks = intervals.count(peak_interval)
        else:
            peak_15_min_incident_time = "N/A"
            peak_15_min_barks = 0
   
        # Placeholders for additional incident details
        avg_barks_per_minute_15 = "7.2"   # Replace with actual calculation when available
        quiet_minutes_15 = "0"            # Replace with actual calculation when available
   
        peak_30_min_incident_time = "11 February 2025 from 21:09:22"  # Replace as needed
        avg_barks_per_minute_30 = "4.2"   # Replace with actual calculation when available
        quiet_minutes_30 = "9"            # Replace with actual calculation when available
   
        # Location details
        nearby_address = "123 Sesame St., New York, NY 10023"
        witnessed_lat_long = "40.77121869483373, -73.98140263086094"
   
        # Loudest bark (using self.highest_db value)
        loudest_bark = f"{self.highest_db:.2f}"
   
        # Prepare the session summary as a dictionary; note that "Recording Stop" is now an integer.
        report_summary = {
            "Recording Start": start_time_str,
            "Recording Stop": session_end_str,
            "Latitude": witnessed_lat_long.split(",")[0].strip(),
            "Longitude": witnessed_lat_long.split(",")[1].strip(),
            "Total Barks Recorded": self.bark_count,
            "Average Barks per Hour": f"{avg_barks_per_hour:.2f}",
            "Loudest Bark": loudest_bark,
            "Peak 15 Min Incident": peak_15_min_incident_time,
            "Average Barks per Minute 15": avg_barks_per_minute_15,
            "Quiet Minutes 15": quiet_minutes_15,
            "Peak 30 Min Incident": peak_30_min_incident_time,
            "Average Barks per Minute 30": avg_barks_per_minute_30,
            "Quiet Minutes 30": quiet_minutes_30,
            "Nearby Address": nearby_address
        }
   
        # Compute Quarter-Hour Breakdown for each hour (0-23)
        counts = {hour: [0, 0, 0, 0] for hour in range(24)}
        for t in self.bark_timestamps:
            # Convert relative timestamp to an absolute event time
            event_time = datetime.fromtimestamp(self.start_time + t)
            hour = event_time.hour
            minute = event_time.minute
            if minute < 15:
                counts[hour][0] += 1
            elif minute < 30:
                counts[hour][1] += 1
            elif minute < 45:
                counts[hour][2] += 1
            else:
                counts[hour][3] += 1
   
        # Write the CSV file
        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
   
            # Write the session summary section
            writer.writerow(["Session Summary"])
            for key, value in report_summary.items():
                writer.writerow([key, value])
   
            # Blank row as separator
            writer.writerow([])
   
            # Write header for the Quarter-Hour Breakdown table
            writer.writerow(["Hour", "15", "30", "45", "60"])
   
            # Write counts for each hour (0 through 23)
            for hour in range(24):
                row = [hour] + counts[hour]
                writer.writerow(row)
   
        print(f"Report saved to {csv_path}")
    
    def on_pdf_selected(self, event):
        """Handles selection of a PDF file and opens it."""
        selection = event.widget.curselection()
        if not selection:
            return  

        selected_index = selection[0]
        selected_file = self.pdf_listbox.get(selected_index)
        file_path = os.path.join(pdf_reports_dir, selected_file)

        print(f"Opening PDF: {file_path}")  # ✅ Debugging message

        try:
            subprocess.run(["xdg-open", file_path], check=True)  # ✅ Opens the PDF
        except Exception as e:
            print(f"Error opening PDF: {e}")
    
    def generate_bark_chart(self):
        """Creates a chart of barking events over time."""
        if not self.bark_timestamps:
            print("? No bark data available for chart.")
            return None  

        timestamps = [self.start_time + ts for ts in self.bark_timestamps]
        frequencies = [analyze_frequency(self.audio_buffer, SAMPLERATE) for _ in self.bark_timestamps]

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, frequencies, 'o-', label="Bark Frequency")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Barking Frequency Over Time")
        plt.legend()
        plt.grid()

        chart_path = os.path.join(pdf_reports_dir, "bark_chart.png")
        plt.savefig(chart_path)
        plt.close()

        return chart_path  
            
    def generate_pdf_report(self):
         # Get the selected CSV file from the listbox.
        selection = self.file_listbox.curselection()
        if not selection:
            tk.messagebox.showinfo("Info", "Please select a CSV file to generate a PDF report.")
            return

        selected_index = selection[0]
        selected_file = self.file_listbox.get(selected_index)
        csv_path = os.path.join(reports_dir, selected_file)

        # Parse the CSV file.
        session_summary = {}
        table_header = []
        table_data = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)

        # Find the Session Summary section.
        i = 0
        while i < len(lines) and (not lines[i] or lines[i][0].strip() != "Session Summary"):
            i += 1

        if i >= len(lines) or not lines[i] or lines[i][0].strip() != "Session Summary":
            tk.messagebox.showerror("Error", "Session Summary not found in CSV.")
            return

        # Skip the "Session Summary" marker.
        i += 1

        # Read key-value pairs until an empty row.
        while i < len(lines) and any(cell.strip() for cell in lines[i]):
            if len(lines[i]) >= 2:
                key = lines[i][0].strip()
                value = lines[i][1].strip()
                session_summary[key] = value
            i += 1

        # Skip blank rows.
        while i < len(lines) and not any(cell.strip() for cell in lines[i]):
            i += 1

        # The next row should be the table header.
        if i < len(lines):
            table_header = lines[i]
            i += 1
        else:
            tk.messagebox.showerror("Error", "Quarter-hour breakdown table not found in CSV.")
            return

        # Read the table rows.
        while i < len(lines) and lines[i]:
            # Ensure we have at least as many columns as the header.
            if len(lines[i]) >= len(table_header):
                table_data.append(lines[i][:len(table_header)])
            i += 1

        # Retrieve required variables from the session summary.
        start_time_str = session_summary.get("Recording Start", "N/A")
        session_end_str = session_summary.get("Recording Stop", "N/A")
        latitude = session_summary.get("Latitude", "N/A")
        longitude = session_summary.get("Longitude", "N/A")

        # Create the PDF.
        pdf = FPDF()
        pdf.add_page()

        # Title.
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Daily Summary Report", ln=True, align="C")
        pdf.ln(3)

        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, "Note: The bark number is the amount of recordings made from barks.", ln=True, align="C")
        pdf.ln(5)

        # Session details.
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Recording Start {start_time_str}", ln=True)
        pdf.cell(0, 8, f"Recording Stop {session_end_str}", ln=True)
        pdf.ln(3)

        # Location details.
        pdf.cell(0, 8, f"Latitude {latitude}", ln=True)
        pdf.cell(0, 8, f"Longitude {longitude}", ln=True)
        pdf.ln(5)

        # Summary date line (using a placeholder as in your sample).
        pdf.cell(0, 8, "4-Mar-25", ln=True, align="C")
        pdf.ln(3)

        # --- Add the table from the CSV with conditional cell coloring ---
        pdf.set_font("Arial", "B", 12)
        # Use 70% of the available width for the table.
        table_width = (pdf.w - 20) * 0.7
        # Calculate the starting x position to center the table.
        start_x = (pdf.w - table_width) / 2
        # Determine column width based on the table width.
        col_width = table_width / len(table_header)

        # Print header row (unchanged)
        pdf.set_x(start_x)
        for header_cell in table_header:
            pdf.cell(col_width, 10, header_cell, border=1, align="C")
        pdf.ln()

        # Print table data rows with color for each cell (except the first "Hour" column)
        pdf.set_font("Arial", "", 12)
        for row in table_data:
            pdf.set_x(start_x)
            for i, cell in enumerate(row):
                # For the first column ("Hour"), no color fill
                if i == 0:
                    pdf.cell(col_width, 10, cell, border=1, align="C")
                else:
                    try:
                        val = int(cell)
                    except ValueError:
                        val = 0
                    if val == 0:
                        pdf.set_fill_color(0, 255, 0)    # green
                    elif 1 <= val <= 14:
                        pdf.set_fill_color(255, 255, 0)  # yellow
                    else:
                        pdf.set_fill_color(255, 0, 0)    # red
                    pdf.cell(col_width, 10, cell, border=1, align="C", fill=True)
            pdf.ln()



        # Footer with creation details.
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, "Created by Bark Witness (TM) March 7 2025 at 14:33", ln=True, align="C")
        pdf.ln(3)
        pdf.cell(0, 8, "Barks Recorded per Quarter Hour", ln=True, align="C")

        # Save the PDF report.
        pdf_filename = os.path.splitext(selected_file)[0] + ".pdf"
        pdf_full_path = os.path.join(pdf_reports_dir, pdf_filename)
        try:
            pdf.output(pdf_full_path)
            tk.messagebox.showinfo("Success", f"Daily Summary PDF generated at:\n{pdf_full_path}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not generate PDF: {e}")
      
    def create_playback_page(self, parent):
        """Creates the playback page with audio file list and playback controls."""
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
    
    def create_pdf_reports_page(self, parent):
        """Creates the PDF Reports page with a list of available PDF files."""
        frame = ctk.CTkFrame(parent)  
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        label = ctk.CTkLabel(frame, text="PDF Reports", font=("Arial", 20))
        label.pack(pady=10, anchor="w")  

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
        
    def refresh_pdf_list(self):
        """Refreshes the list of available PDF reports."""
        if not hasattr(self, "pdf_listbox"):
            print("Error: pdf_listbox is not initialized yet.")
            return
    
        self.pdf_listbox.delete(0, tk.END)
        pdf_files = [f for f in os.listdir(pdf_reports_dir) if f.endswith(".pdf")]

        for file in pdf_files:
            self.pdf_listbox.insert(tk.END, file)
    
        print("PDF file list refreshed")

    def refresh_audio_list(self):
        """Refreshes the list of available audio files."""
        if not hasattr(self, "audio_listbox"):
            print("Error: audio_listbox is not initialized yet.")
            return
        
        self.audio_listbox.delete(0, tk.END)
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

        for file in audio_files:
            self.audio_listbox.insert(tk.END, file)
        
        print("Audio file list refreshed")

    def on_audio_selected(self, event):
        """Handles selection of an audio file from the playback list."""
        selection = event.widget.curselection()
        if not selection:
            return  

        selected_index = selection[0]
        selected_file = self.audio_listbox.get(selected_index)
        print(f"Selected audio file: {selected_file}")

    def play_selected_audio(self):
        """Plays the selected audio file with correct formatting and stops previous playback."""
        selection = self.audio_listbox.curselection()
        if not selection:
            tk.messagebox.showinfo("Info", "Please select an audio file to play.")
            return

        selected_index = selection[0]
        selected_file = self.audio_listbox.get(selected_index)

        if not selected_file.endswith("_s16le.wav"):
            tk.messagebox.showerror("Error", "Please select a valid converted file (_s16le.wav).")
            return

        file_path = os.path.join(audio_dir, selected_file)

        if not os.path.exists(file_path):
            tk.messagebox.showerror("Error", "Selected audio file not found.")
            return

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
                ["aplay", "-D", "hw:0,0", "-f", "S16_LE", "-r", "48000", "-c", "2", file_path]
            )
            print(f"?? Playing {file_path} on hw:0,0 with S16_LE format.")
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

    def create_intro_page(self, parent):
        label = ctk.CTkLabel(
            parent,
            text="Introduction to Bark Witness",
            font=("Arial", 20), wraplength=600
        )
        label.pack(pady=20)

    def create_settings_page(self, parent):
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

        label, output_data = self.classify_audio(np.array(self.audio_buffer[-self.buffer_size:]))

        dog_index = labels.index("Dog Barking")
        dog_confidence = output_data[0][dog_index]

        if label == "Dog Barking" and dog_confidence >= self.threshold:
            self.bark_count += 1
            self.bark_timestamps.append(time.time() - self.start_time)
            print(f"Bark detected! Confidence: {dog_confidence:.2f}")

        self.after(0, lambda: self.detection_label.configure(
            text=f"Barks This Session: {self.bark_count}\n(Dog Confidence: {dog_confidence:.2f})"
        ))
   
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
        """ Save the bark audio with both pre-bark and post-bark data """
        print("Recording post-bark audio...")
        duration = 5  
        post_bark_audio = sd.rec(int(48000 * duration), samplerate=48000, channels=2, dtype='int16', device="hw:1,0")
        sd.wait()

        post_bark_audio = post_bark_audio.flatten()
        full_audio = np.concatenate((np.array(pre_bark_audio)[-self.buffer_size:], post_bark_audio))

        if len(full_audio) == 0:
            print("ERROR: No audio captured! Check audio buffer.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.recordings_folder, f"bark_{timestamp}.wav")

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2) 
            wf.setsampwidth(2) 
            wf.setframerate(48000)
            wf.writeframes(full_audio.tobytes())

        print(f"? Bark audio saved: {filename}")

    def generate_csv_report(self): 
        """Generates a CSV report for recorded dog barking events, including GPS coordinates."""

        if not hasattr(self, "start_time") or self.start_time is None:
            print("? Error: No valid start time recorded. Cannot generate report.")
            return

        reports_dir = os.path.join(script_dir, "CSV_Files")
        os.makedirs(reports_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{date_str}_report.csv"
        csv_path = os.path.join(reports_dir, filename)

        start_time_str = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = time.time() - self.start_time
        hours_elapsed = elapsed_time / 3600
        avg_barks_per_hour = self.bark_count / hours_elapsed if hours_elapsed > 0 else 0

        lat, lon, utc = self.gps_data.get("lat", "N/A"), self.gps_data.get("lon", "N/A"), self.gps_data.get("utc", "N/A")
        location = f"Lat: {lat}, Lon: {lon}" if lat != "N/A" else "GPS Coordinates Not Available"

        peak_incident_start = "N/A"
        peak_15_min_barks = 0
        if self.bark_timestamps:
            intervals = [t // 900 for t in self.bark_timestamps] 
            peak_15_min_interval = max(set(intervals), key=intervals.count)
            peak_15_min_barks = intervals.count(peak_15_min_interval)
            peak_incident_start = datetime.fromtimestamp(
                self.start_time + peak_15_min_interval * 900
            ).strftime("%H:%M:%S")

        report_data = {
            "Start Time": start_time_str,
            "Elapsed Time": f"{int(elapsed_time // 3600):02}:{int((elapsed_time % 3600) // 60):02}:{int(elapsed_time % 60):02}",
            "Total Barks Detected": self.bark_count,
            "Average Barks per Hour": f"{avg_barks_per_hour:.2f}",
            "Peak Incident Start": peak_incident_start,
            "Peak 15 Min Incident": peak_15_min_barks,
            "Highest dB Recorded": f"{self.highest_db:.2f} dB",
            "Location": location,
            "Latitude": lat,
            "Longitude": lon,
            "UTC Time": utc,
            "Lowest Confidence Threshold": f"{self.lowest_threshold:.2f}%"
        }

        try:
            with open(csv_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=report_data.keys())
                writer.writeheader()
                writer.writerow(report_data)

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
                headers = next(reader, None)

                if not headers:
                    error_label = ctk.CTkLabel(csv_window, text="?? Empty or corrupt CSV file.", font=("Arial", 14))
                    error_label.pack(pady=10)
                    return

                for row in reader:
                    frame = ctk.CTkFrame(csv_window)
                    frame.pack(fill="x", padx=10, pady=5)

                    for key, value in zip(headers, row):
                        key_label = ctk.CTkLabel(frame, text=f"{key}:", font=("Arial", 14), anchor="w")
                        key_label.pack(side="left", padx=5)

                        value_label = ctk.CTkLabel(frame, text=value, font=("Arial", 14), anchor="w")
                        value_label.pack(side="left", fill="x", expand=True)

        except Exception as e:
            error_label = ctk.CTkLabel(csv_window, text=f"Error reading file: {str(e)}", font=("Arial", 14))
            error_label.pack(pady=10)

    def generate_pdf_report(self):
        """Converts CSV Files from "/home/bark1/CSV_Files/" into properly formatted PDF documents."""
        os.makedirs(pdf_reports_dir, exist_ok=True) 
        csv_files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]

        if not csv_files:
            print("? No CSV Files found in /home/bark1/CSV_Files/")
            tk.messagebox.showinfo("Info", "No CSV Files found to convert.")
            return

        for csv_file in csv_files:
            csv_path = os.path.join(reports_dir, csv_file)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.cell(0, 10, f"Report: {csv_file}", ln=True, align="C")
            pdf.ln(5)

            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)

                    if not headers:
                        print(f"? Skipping empty CSV file: {csv_file}")
                        continue

                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(0, 10, ', '.join(headers), ln=True)

                    pdf.set_font("Arial", size=10)
                    for row in reader:
                        line = ', '.join(row)
                        pdf.multi_cell(0, 10, txt=line)

                pdf_filename = os.path.splitext(csv_file)[0] + ".pdf"
                pdf_path = os.path.join(pdf_reports_dir, pdf_filename)
                pdf.output(pdf_path)

                print(f"? PDF Report saved: {pdf_path}")

            except Exception as e:
                print(f"? Error processing CSV file {csv_file}: {e}")

        tk.messagebox.showinfo("Success", "PDF_reports successfully generated in /home/bark1/PDF_Reports/!")

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        csv_files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]

        for file in csv_files:
            self.file_listbox.insert(tk.END, file)
        print('CSV file list refreshed')

    def refresh_audio_list(self):
        self.audio_listbox.delete(0, tk.END)
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]
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
            os.path.join(script_dir, "Recordings")
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








