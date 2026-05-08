import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import datetime
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from lib import acoustics
# import os, sys
# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if root_path not in sys.path:
#     sys.path.insert(0, root_path)
# # import parent folder to access modules
from Processing import reverberation_time as rt

base_folder = Path(__file__).parent.parent
rir_file = base_folder / "RIRs" / "pos_test_probe" / "rir_A_20260505_163748.wav"

print(base_folder)


rir,fs = sf.read(rir_file)  # Read the RIR from the file


rt30 = rt.calculate_reverberation_time(rir = rir,
    fs = fs,
    method = "T20",
    plot = True,
)




plt.figure(figsize=(12, 6))

plt.plot(rir)
plt.title('Estimated Room Impulse Response')
plt.show()