import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from py_octave_band import octavefilter
from lib import acoustics
import soundfile as sf

rir = "/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/Experiments/Measuring_tests/RIRs/pos_pos_test1/RIR_20260422_114447.wav"
rir = sf.read(rir)[0]
rt = acoustics.rt(rir, fs=48000, db=60, db_start=5, db_end=25, plot=True)
print(rt)
plt.show()
