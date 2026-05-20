import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import datetime
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from lib.acoustics import rt

base_folder = Path(__file__).parent.parent
print(base_folder)

# ENTER. FULL PATH and FILE NAME of the RIR you want to analyze here:
rir_file = "none"  

# Check if rir_file is defined and is a valid file path
try:
    if not Path(rir_file).is_file():
        raise FileNotFoundError("Invalid RIR file path")
except (NameError, FileNotFoundError, TypeError):
    from tkinter import Tk
    from tkinter import filedialog

    root = Tk()
    root.withdraw()  # Hide the root window
    rir_file = filedialog.askopenfilename(
        title="Select RIR file", initialdir=base_folder, filetypes=[("WAV files", "*.wav")])
    if not rir_file:
        print("No file selected. Exiting.")
        exit()  

rir,fs = sf.read(rir_file)  # Read the RIR from the file
T30 = rt(rir=rir, fs=fs, method="T30", plot=True)

