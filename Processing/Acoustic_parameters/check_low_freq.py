from py_octave_band import octavefilter
from lib import acoustics, sweep
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
import os
from pathlib import Path
#for exported plots
# import scienceplots
# plt.style.use(["science", "ieee", "grid", "bright"])
# %matplotlib widget


class Measurement:
    def __init__(self, folder_name, measurement_prefix, microphone_identifier, fs):
        self.param_labels = [
            "$T_{20}$ [s]", "EDT [s]", "$C_{80}$ [dB]", "$D_{50}$ [dB]", "$T_s$ [s]", "$E$ [dB]", "DRR"]
        self.folder_name = folder_name
        self.meas_name = measurement_prefix
        self.microphone_identifier = microphone_identifier
        self.fs = fs
        self.freqbands = None
        self.broadband_params = None
        self.subband_params = None
        self.rirs = []

def load_measurements(folder, measurement_prefix, microphone_identifier):
    measurements = []

    for i in measurement_prefix:
        # Find the subfolder for each measurement prefix
        subfolder = Path(folder) / f"{i}"
        print(f"Processing folder: {subfolder}")

        if not subfolder.exists():
            print(f"Folder {subfolder} does not exist.")
            continue

        rirs = []
        fs = None  # Initialize sampling rate

        for file in os.listdir(subfolder):
            # Only pick files containing the microphone_identifier
            if any(mic in file for mic in microphone_identifier):
                # Load the file
                rir, fs = sf.read(subfolder / file)

                print("read file {}/{}".format(subfolder , file))
                # concatenate strings subforlder and file
                rirs.append(rir)

        # Skip if no valid RIRs were found
        if not rirs:
            print(f"No valid RIRs found in {subfolder}")
            continue

        # Create a Measurement object and calculate parameters
        measurement = Measurement(subfolder, measurement_prefix, microphone_identifier, fs)
        measurement.rirs = rirs
        measurements.append(measurement)

    return measurements


measurement_prefix = ["A1","A2","A3","B1","B2","B3","C1","D2"]
microphone_identifier = ["E"]

fs = 48_000

# enter subfolders and display file that contain microphone_identifiers in the name

current_dir = Path.cwd()
folder_up = current_dir.parents[1]
# folder RIR two folders up
folder = folder_up / "RIRvisual"


measurements = load_measurements(folder,measurement_prefix,microphone_identifier)

for mnr,meax in enumerate(measurements):
    
    # plot frequency fft of rir from 0 to 200 Hz
    # display the frequency resolution of the fft
    #freqeuncy resolution = fs/N

    N = len(meax.rirs[0])
    freq_res = fs/N
    print(f"Frequency resolution: {freq_res} Hz")

    f_limits = [0, 200]
    f_limits = [int(f_limit/freq_res) for f_limit in f_limits]
    f_limits = [f_limit if f_limit < N else N for f_limit in f_limits]
    f_limits = [f_limit if f_limit > 0 else 1 for f_limit in f_limits]
    print(f_limits)
    f = np.linspace(0, fs, N)
    plt.figure()

    plt.scatter(np.abs(np.fft.fft(meax.rirs[0])[:200]))
    plt.title(f"frequency response at position: {meax.folder_name.name}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.show()



   


# %%
