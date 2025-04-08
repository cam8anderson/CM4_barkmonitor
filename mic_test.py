import os
import pyaudio
import wave
import threading
import datetime
from pydub import AudioSegment  # For converting WAV to MP3

# Parameters
CHUNK = 1024             # Number of frames per buffer
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1             # Mono (change to 2 for stereo)
RATE = 48000             # Sample rate
WAVE_OUTPUT_FILENAME = "output.wav"

OUTPUT_DIR = "BARKS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
WAVE_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, f"output_{timestamp}.wav")

# Initialize PyAudio
p = pyaudio.PyAudio()

def find_device(p, device_str):
    """Search for an input device whose name contains device_str."""
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if device_str in info.get('name', ''):
            return i
    return None

# Attempt to automatically find the device "hw:1,0"
device_str = "hw:1,0"
device_index = find_device(p, device_str)
if device_index is None:
    print(f"Device '{device_str}' not found automatically.")
    device_index = int(input("Enter the device index manually: "))
else:
    print(f"Using device index {device_index} for '{device_str}'.")

# Open stream using the selected input device
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_index)

frames = []  # List to store audio frames
recording = True  # Flag to control recording

def wait_for_stop():
    """Wait for user input to stop recording."""
    global recording
    input("Press Enter to stop recording...\n")
    recording = False

# Start a thread that waits for the user to press Enter
stop_thread = threading.Thread(target=wait_for_stop)
stop_thread.start()

print("Recording...")

# Record in chunks until recording is stopped
while recording:
    try:
        data = stream.read(CHUNK)
        frames.append(data)
    except Exception as e:
        print("Error while recording:", e)
        break

print("Finished recording.")

# Clean up the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded frames as a WAV file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Recording saved to '{WAVE_OUTPUT_FILENAME}'.")

# Convert the WAV file to MP3 using pydub
mp3_filename = WAVE_OUTPUT_FILENAME.replace(".wav", ".mp3")
audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
audio.export(mp3_filename, format="mp3")
print(f"Recording converted to MP3 and saved to '{mp3_filename}'.")
