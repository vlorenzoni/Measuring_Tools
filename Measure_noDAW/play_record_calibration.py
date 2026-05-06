import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import datetime

# Example usage
stimulus_file = "/Users/valeriolorenzoni/Dropbox/My_ESAT/Experiments/GroepT-Jan2025/Measurements/calibration/input4calibration_groepT_duration15_silAtEnd10_repetitions2.wav"  # Replace with your .wav file path


# ENTER POSITION 
positions = "calibration"  # Folder names for mic positions


base_folder = Path('/Users/valeriolorenzoni/Dropbox/My_ESAT/Experiments/GroepT-Jan2025/Measurements/recordings/')
   # Base folder to store recordings
mic_channels = 5  # Number of microphones in the array
mic_names = ['7','8','9','10','11']  # Names of the microphones


# Load the stimulus file
stimulus, fs = sf.read(stimulus_file)


original =  48000
recorded =  57310

measured_delay = recorded - original
print(f"Measured delay: {measured_delay} samples")
print(f"Measured delay: {measured_delay / fs} seconds")





#plot the stimulus
import matplotlib.pyplot as plt
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
    input_mapping=list(range(7, 12)),
    output_mapping=[1],
    device=device_id_or_name  
)

sd.wait()        
    # Generate filename and save individual microphone channels

# Save the recording
for mic_idx in range(mic_channels):
    # include position and timestamp in filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the folder if it does not exist

    position_folder = base_folder / f"pos_{positions}"
    if not position_folder.exists():
        position_folder.mkdir(parents=True, exist_ok=True)
    #add time stamp to file name
    mic_file_path = position_folder / f"mic_{mic_names[mic_idx]}_{timestamp}.wav"
    print(mic_file_path)
    sf.write(mic_file_path, recording[:, mic_idx], fs)
    #display full filename with folder
    


   
   
