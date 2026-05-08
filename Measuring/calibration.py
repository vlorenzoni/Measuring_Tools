from signal import signal
from turtle import position

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt


# This script is used to measure the latency of the whole measuring system, including the soundcard, the cables, and the microphones. To do this. hsortcut the speaked directly back into the input channel. Then play an impulsive signal and record it. By comparing the original stimulus with the recorded signal, we can calculate the latency in samples. This latency can then be used as an offset in the RIR estimation to correct for any delays in the system, ensuring that the direct sound and early reflections are accurately represented in the estimated RIR.


fs = 48000  # Sample rate

mic_channel = 9  # Use one single channel. Specify the number from soundcard or from Totalmix.

base_folder = Path(__file__).parent.parent

peak_position = fs  # Position of the pulse in the stimulus (1 second at 48 kHz)
stimulus = np.zeros(2 * fs)  
stimulus[peak_position] = 1 # Pulse at 1 second  at 48 kHz



plt.plot(stimulus)
plt.show()



# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Set the desired device (either name or ID)
device_id_or_name = "Fireface UFX+"  # Replace with the device ID or name you want to use
print(f"Using device: {device_id_or_name}")



recording = sd.playrec(
    stimulus,
    samplerate=fs,
    input_mapping= mic_channel,
    output_mapping=[1],
    device= device_id_or_name  # to try on laptop use: (0,1)
)

sd.wait()        



# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Set the desired device (either name or ID)
device_id_or_name = "Fireface UFX+"  # Replace with the device ID or name you want to use
print(f"Using device: {device_id_or_name}")

recording = sd.playrec(
    stimulus,
    samplerate=fs,
    input_mapping=mic_channel,
    output_mapping=[1],
    device=device_id_or_name  
)

recording = recording.flatten()  
sd.wait()        
    # Generate filename and save individual microphone channels

recording_folder = base_folder / "Calibration"
if not recording_folder.exists():
    recording_folder.mkdir(parents=True, exist_ok=True)


# include position and timestamp in filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create the folder if it does not exist

recording_folder
#add time stamp to file name
mic_file_path = recording_folder / f"mic_{mic_channel}_{timestamp}.wav"
print(mic_file_path)
sf.write(mic_file_path, recording, fs)
#display full filename with folder


original_pulse_position = peak_position
recording_delay_samples = np.argmax(recording)  # Find the index of the maximum value in the recording

offset_samples = recording_delay_samples - original_pulse_position
print(f"Estimated latency (offset) in samples: {offset_samples}")


