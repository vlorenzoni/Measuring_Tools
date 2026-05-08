import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import datetime
from scipy.signal import fftconvolve
from lib import sweeps_methods as swp
import matplotlib.pyplot as plt

SAMPLE_RATE = 48000                 # Hz    - sample rate for audio playback
BLOCK_LENGTH = 1024                 # samples - Only used for the spectrogram visualization, not for the actual measurement. Adjust as needed for better time/frequency resolution in the spectrogram.
F_START = 1                         # Hz    - sweep start frequency
F_FINAL = SAMPLE_RATE // 2          # Hz    - sweep end frequency (Nyquist)
T_SWEEP = 4                         # sec   - duration of the sine sweep
T_IDLE  = 2                         # sec   - silence appended after the sweep
VOLUME = 0.8                        # Linear gain for the sweep signal (0.0 to 1.0)
OFFSET = 0                          # samples - integer from calibration,  to be applied to the RIR to correct for latency of the whole mesuring system -- ONLY APPLIED IF THE CASUALITY PARAMETER IN THE RIR ESTIMATION FUNCTION IS SET TO TRUE, IF SET TO FALSE, THE RIR WILL BE ESTIMATED WITHOUT ANY TIME SHIFT AND  OFFSET, DEFAULT OFFSET  = 0 samples


base_folder = Path(__file__).parent.parent
position = "test_probe"

mic_channels = 6  # Number of microphones in the array
mic_names = ['A', 'B', 'C', 'D', 'E', 'F']  # Names of the microphones

sweep, inverse = swp.ess_gen_farina(
        F_START, F_FINAL, T_SWEEP, T_IDLE, SAMPLE_RATE,
        fade_in=128, cut_zerocross=True, sweep_gain=VOLUME
    )

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Chosen device for recording and playback
device_id_or_name = "Fireface UFX+"  # Replace with the device ID or name you want to use
print(f"Using device: {device_id_or_name}")

recording = sd.playrec(
    sweep,
    samplerate=SAMPLE_RATE,
    input_mapping=list(range(31, 37)), # Correspond to 6 probe cables on the first 6 channel of RME 12mic card (under the UFX+), adjust if needed
    output_mapping=[1],
    device=device_id_or_name  
)

sd.wait()        


# Save the recording
for mic_idx in range(mic_channels):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_folder = base_folder / "recordings" / f"pos_{position}"
    # Create the folder if it does not exist
    if not recording_folder.exists():
        recording_folder.mkdir(parents=True, exist_ok=True)

    mic_file_name = recording_folder / f"mic_{mic_names[mic_idx]}_{timestamp}.wav"
    print(mic_file_name)     
    
    sf.write(mic_file_name, recording[:, mic_idx], SAMPLE_RATE)
    
    rir = swp.ess_parse_farina(
        recording[:, mic_idx], inverse, T_SWEEP, T_IDLE, SAMPLE_RATE, offset=OFFSET, causality=False
    )

    rir_folder = base_folder / "RIRs" / f"pos_{position}"
    if not rir_folder.exists():
        rir_folder.mkdir(parents=True, exist_ok=True)   
    rir_file_name = rir_folder / f"rir_mic_{mic_names[mic_idx]}_{timestamp}.wav"
    print(rir_file_name)
    sf.write(rir_file_name, rir, SAMPLE_RATE)   

   

# CHECK MEASUREMENTS NOISE on last microphone

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(recording[:, mic_idx])
plt.title("Recorded Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(recording))
plt.grid()  
plt.subplot(2, 1, 2)
plt.specgram(recording[:, mic_idx], Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
plt.title("Spectrogram of Recorded Signal")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.tight_layout()


## CHECK RIR from last microphone

plt.figure(figsize=(12, 6))
plt.plot(rir)
plt.title("Estimated RIR")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(rir))
plt.grid()
plt.show()
