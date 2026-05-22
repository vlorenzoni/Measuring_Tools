import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import datetime
from scipy.signal import fftconvolve
from lib import sweeps_methods as swp
import matplotlib.pyplot as plt
from lib.measurement_quality import check_recording, check_rir


SAMPLE_RATE = 48000                 # Hz    - sample rate for audio playback
F_START = 1                         # Hz    - sweep start frequency
F_FINAL = SAMPLE_RATE // 2          # Hz    - sweep end frequency (Nyquist)
T_SWEEP = 4                         # sec   - duration of the sine sweep
T_IDLE  = 3                         # sec   - silence appended after the sweep
VOLUME = 1                          # int   - gain of the sweep

# ENTER POSITION 
position = "test_1mic"    # Name of the position, used for folder naming when      saving recordings and RIRs. Adjust as needed for your measurement setup.

# SPECIFY MICROPHONE CHANNEL
mic_channel = 9  # Microphone channel number from soundcard or from totalmix.

# SPECIFY OFFSET
OFFSET = 9309                          # samples - MEASURED INTEGER FROM CALIBRATION ,  to be applied to the RIR to correct for latency of the whole mesuring system -- necessary to estimate a correct direct sound and early reflections in the RIR, ONLY APPLIED causality = TRUE , default  = 0 samples.  To measure it shortcut the system and use Calibration.py routing.



base_folder = Path(__file__).parent.parent

# Generate the sweep signal
sweep,inverse_sweep = swp.ess_gen_farina(
        F_START, F_FINAL, T_SWEEP, T_IDLE, SAMPLE_RATE,
        fade_in=128, cut_zerocross=True, sweep_gain=VOLUME
    )

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Set the desired device (either name or ID)
device_id_or_name = "Fireface UFX+"  # Replace with the device ID or name you want to use
print(f"Using device: {device_id_or_name}")

recording = sd.playrec(
    sweep,
    samplerate=SAMPLE_RATE,
    input_mapping= mic_channel,
    output_mapping=[1],
    device= device_id_or_name  # to try on laptop use: (0,1)
)

sd.wait()        

# Save recording
recording = recording.flatten()  # Flatten to 1D if it's a single channel recording
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
recording_folder = base_folder / "Recordings" / f"pos_{position}"
if not recording_folder.exists():
    recording_folder.mkdir(parents=True, exist_ok=True)  
recording_file_path = recording_folder / f"recording_{timestamp}.wav"
print(recording_file_path)
sf.write(recording_file_path, recording.astype(np.float32), SAMPLE_RATE)   

# Calculate and save rir
rir = swp.ess_parse_farina(
    recording, inverse_sweep, T_SWEEP, T_IDLE, SAMPLE_RATE, offset=OFFSET, causality=True
)


rir_folder = base_folder / "RIRs" / f"pos_{position}"
if not rir_folder.exists():
    rir_folder.mkdir(parents=True, exist_ok=True)   
rir_file_path = rir_folder / f"rir_{timestamp}.wav"
print(rir_file_path)
sf.write(rir_file_path, rir.astype(np.float32), SAMPLE_RATE)   


# CHECK MEASUREMENTS NOISE

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(recording)
plt.title("Recorded Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(recording))
plt.grid()  
plt.subplot(2, 1, 2)
plt.specgram(recording, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
plt.title("Spectrogram of Recorded Signal")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.tight_layout()


## CHECK RIR

plt.figure(figsize=(12, 6))
plt.plot(rir)
plt.title("Estimated RIR")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(rir))
plt.grid()
plt.show()

## Enable quality checks
check_recording(recording, sweep, SAMPLE_RATE, T_SWEEP, T_IDLE, mic_name="mic")
check_rir(rir, SAMPLE_RATE, mic_name="mic")  
