"""
Reverberation Time Calculator from Room Impulse Response (RIR)
================================================================
Calculates RT60, RT30, and RT20 per octave/third-octave frequency band,
then averages the result in seconds.

Methods:
  - Schroeder backward integration (ISO 3382-1 compliant)
  - Least-squares linear regression on the energy decay curve (EDC)

Usage:
  python reverberation_time.py                        # runs built-in demo with synthetic RIR
  python reverberation_time.py --wav path/to/rir.wav  # load a real RIR from a WAV file
"""

import argparse
import numpy as np
from scipy.signal import butter, sosfilt
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# --------------------------------------------------------------------------- #
#  Optional imports (graceful degradation)                                     #
# --------------------------------------------------------------------------- #
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[INFO] matplotlib not found – plots will be skipped.\n")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# --------------------------------------------------------------------------- #
#  Octave-band centre frequencies (ISO 266)                                    #
# --------------------------------------------------------------------------- #
OCTAVE_BANDS_HZ = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
THIRD_OCTAVE_BANDS_HZ = [
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000,
]


# --------------------------------------------------------------------------- #
#  Bandpass filter                                                              #
# --------------------------------------------------------------------------- #
def bandpass_filter(signal: np.ndarray, fc: float, fs: float,
                    order: int = 6, fraction: int = 1) -> np.ndarray:
    """
    Apply an octave (fraction=1) or 1/3-octave (fraction=3) bandpass filter.

    Parameters
    ----------
    signal  : 1-D impulse response (time domain)
    fc      : centre frequency in Hz
    fs      : sample rate in Hz
    order   : Butterworth filter order (default 6)
    fraction: 1 = octave bands, 3 = third-octave bands

    Returns
    -------
    Filtered signal (same length as input).
    """
    factor = 2 ** (1.0 / (2 * fraction))
    fl = fc / factor
    fh = fc * factor
    nyq = fs / 2.0
    # Guard against out-of-range frequencies
    if fl <= 0 or fh >= nyq:
        return np.zeros_like(signal)
    sos = butter(order, [fl / nyq, fh / nyq], btype="bandpass", output="sos")
    return sosfilt(sos, signal)


# --------------------------------------------------------------------------- #
#  Energy Decay Curve (Schroeder backward integration)                         #
# --------------------------------------------------------------------------- #
def energy_decay_curve(h: np.ndarray) -> np.ndarray:
    """
    Compute the normalised Energy Decay Curve (EDC) in dB via Schroeder
    backward integration.

    EDC(t) = 10 * log10( integral_{t}^{inf} h²(τ) dτ  /  integral_{0}^{inf} h²(τ) dτ )

    Returns an array of the same length; 0 dB at t=0.
    """
    energy = h ** 2
    # Backward cumulative sum then flip back
    edc_linear = np.cumsum(energy[::-1])[::-1]
    # Avoid log of zero
    edc_linear = np.maximum(edc_linear, 1e-20 * edc_linear[0])
    edc_db = 10.0 * np.log10(edc_linear / edc_linear[0])
    return edc_db


# --------------------------------------------------------------------------- #
#  Reverberation time via linear regression on the EDC                        #
# --------------------------------------------------------------------------- #
def estimate_rt(edc_db: np.ndarray, fs: float,
                method: str = "T20") -> tuple[float, float, np.ndarray]:
    """
    Estimate reverberation time from an EDC (in dB) using least-squares
    linear regression over the evaluation window defined by `method`.

    Parameters
    ----------
    edc_db  : Energy Decay Curve in dB (length N)
    fs      : sample rate in Hz
    method  : 'T20' | 'T30' | 'T60'

    Returns
    -------
    rt_sec   : estimated RT in seconds (extrapolated to −60 dB)
    r_value  : Pearson r of the regression (quality indicator, –1 to 0)
    fit_line : regression line evaluated at every sample point (dB)
    """
    method = method.upper()
    windows = {
        "T20": (-5.0, -25.0),
        "T30": (-5.0, -35.0),
        "T60": (-5.0, -65.0),
    }
    if method not in windows:
        raise ValueError(f"method must be one of {list(windows.keys())}")

    upper_db, lower_db = windows[method]
    n = len(edc_db)
    t = np.arange(n) / fs  # time axis in seconds

    # Find sample indices closest to upper and lower dB limits
    def db_to_idx(db_level):
        diff = np.abs(edc_db - db_level)
        idx = int(np.argmin(diff))
        return idx

    i_upper = db_to_idx(upper_db)
    i_lower = db_to_idx(lower_db)

    if i_lower <= i_upper:
        return np.nan, np.nan, np.full(n, np.nan)

    seg_t = t[i_upper:i_lower]
    seg_edc = edc_db[i_upper:i_lower]

    # Least-squares fit: edc = slope * t + intercept
    coeffs = np.polyfit(seg_t, seg_edc, 1)
    slope, intercept = coeffs

    if slope >= 0:
        return np.nan, np.nan, np.full(n, np.nan)

    # Extrapolate to −60 dB
    rt_sec = -60.0 / slope
    # Pearson correlation coefficient
    seg_fit = np.polyval(coeffs, seg_t)
    ss_res = np.sum((seg_edc - seg_fit) ** 2)
    ss_tot = np.sum((seg_edc - np.mean(seg_edc)) ** 2)
    r_value = -np.sqrt(1.0 - ss_res / max(ss_tot, 1e-30))  # negative (decay)

    fit_line = np.polyval(coeffs, t)
    return rt_sec, r_value, fit_line


# --------------------------------------------------------------------------- #
#  Main public function                                                         #
# --------------------------------------------------------------------------- #
def calculate_reverberation_time(
    rir: np.ndarray,
    fs: float,
    bands: list[float] | None = None,
    fraction: int = 1,
    method: str = "T30",
    plot: bool = True,
) -> dict:
    """
    Calculate reverberation time from a Room Impulse Response.

    Parameters
    ----------
    rir      : 1-D array, the room impulse response (time domain)
    fs       : sample rate in Hz
    bands    : list of centre frequencies in Hz.
               Defaults to octave bands [63 … 8000] Hz.
    fraction : 1 = octave bands, 3 = third-octave bands
    method   : 'T20', 'T30', or 'T60'
    plot     : whether to plot the EDC and regression lines

    Returns
    -------
    result : dict with keys:
        'bands_hz'   – list of centre frequencies used
        'rt_seconds' – RT per band (NaN if estimation failed)
        'r_values'   – regression quality per band
        'rt_mean'    – mean RT across valid bands (seconds)
        'rt_mid'     – mean of 500 Hz + 1 kHz bands (ISO 3382 definition)
    """
    if bands is None:
        bands = OCTAVE_BANDS_HZ

    rir = np.asarray(rir, dtype=float)
    if rir.ndim > 1:
        rir = rir[:, 0]  # take first channel if stereo

    rt_list = []
    r_list = []
    edc_list = []
    fit_list = []

    for fc in bands:
        filtered = bandpass_filter(rir, fc, fs, fraction=fraction)
        edc = energy_decay_curve(filtered)
        rt, r, fit = estimate_rt(edc, fs, method=method)
        rt_list.append(rt)
        r_list.append(r)
        edc_list.append(edc)
        fit_list.append(fit)

    rt_array = np.array(rt_list, dtype=float)
    r_array = np.array(r_list, dtype=float)

    valid_mask = np.isfinite(rt_array)
    rt_mean = float(np.nanmean(rt_array)) if valid_mask.any() else np.nan

    # ISO 3382 mid-frequency average (500 Hz + 1 kHz)
    mid_indices = [i for i, f in enumerate(bands) if f in (500, 1000)]
    rt_mid_vals = rt_array[mid_indices]
    rt_mid = float(np.nanmean(rt_mid_vals)) if len(mid_indices) > 0 else np.nan

    # ------------------------------------------------------------------ #
    #  Console report                                                      #
    # ------------------------------------------------------------------ #
    band_label = "Octave" if fraction == 1 else "1/3-Octave"
    print("=" * 60)
    print(f"  Reverberation Time ({method}) — {band_label} Bands")
    print("=" * 60)
    print(f"  {'Centre Freq (Hz)':>16}   {'RT (s)':>8}   {'r':>6}")
    print("  " + "-" * 40)
    for fc, rt, r in zip(bands, rt_array, r_array):
        rt_str = f"{rt:.3f}" if np.isfinite(rt) else "  N/A "
        r_str  = f"{r:.4f}" if np.isfinite(r)  else "  N/A "
        print(f"  {fc:>16}   {rt_str:>8}   {r_str:>6}")
    print("  " + "-" * 40)
    print(f"  {'Mean RT (all bands)':>16}   {rt_mean:.3f} s")
    print(f"  {'RT_mid (500+1kHz)':>16}   {rt_mid:.3f} s")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Optional plot                                                       #
    # ------------------------------------------------------------------ #
    if plot and MATPLOTLIB_AVAILABLE:
        _plot_edc(bands, edc_list, fit_list, rt_array, fs, method, fraction)

    return {
        "bands_hz":   bands,
        "rt_seconds": rt_array.tolist(),
        "r_values":   r_array.tolist(),
        "rt_mean":    rt_mean,
        "rt_mid":     rt_mid,
    }


# --------------------------------------------------------------------------- #
#  Plotting helper                                                              #
# --------------------------------------------------------------------------- #
def _plot_edc(bands, edc_list, fit_list, rt_array, fs, method, fraction):
    n_bands = len(bands)
    ncols = 4
    nrows = int(np.ceil(n_bands / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    band_label = "Oct" if fraction == 1 else "1/3-Oct"

    for idx, (fc, edc, fit, rt) in enumerate(
            zip(bands, edc_list, fit_list, rt_array)):
        ax = axes[idx]
        t = np.arange(len(edc)) / fs
        ax.plot(t, edc, color="#2196F3", lw=1.2, label="EDC")
        if np.isfinite(rt):
            ax.plot(t, fit, color="#F44336", lw=1.5, ls="--",
                    label=f"{method}={rt:.2f}s")
        ax.axhline(-60, color="#888", lw=0.8, ls=":")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-80, 5)
        ax.set_title(f"{fc} Hz ({band_label})", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Level (dB)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for ax in axes[n_bands:]:
        ax.set_visible(False)

    fig.suptitle(f"Energy Decay Curves — {method}", fontsize=13, y=1.01)
    # plt.savefig("/mnt/user-data/outputs/reverberation_edc_plot.png",
    #             dpi=150, bbox_inches="tight")
    print("\n[INFO] EDC plot saved to reverberation_edc_plot.png")
    plt.show()

def select_wav_file():
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select a WAV file",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )

    # Check if a file was selected
    if not file_path:
        print("No file selected.")
        return None

    # Validate file extension
    if not file_path.lower().endswith('.wav'):
        messagebox.showerror("Invalid File", "Please select a file with a .wav extension.")
        print(f"Error: {file_path} is not a WAV file.")
        return None

    print(f"Selected file: {file_path}")
    return file_path


# --------------------------------------------------------------------------- #
#  CLI entry point                                                              #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Calculate reverberation time from a RIR.")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to a WAV file containing the RIR.")
    parser.add_argument("--fs", type=float, default=48000,
                        help="Sample rate (only used with synthetic demo).")
    parser.add_argument("--rt60", type=float, default=0.8,
                        help="Target T60 for synthetic demo (seconds).")
    parser.add_argument("--method", type=str, default="T30",
                        choices=["T20", "T30", "T60"],
                        help="Evaluation method (default: T30).")
    parser.add_argument("--third-octave", action="store_true",
                        help="Use 1/3-octave bands instead of octave bands.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Suppress EDC plots.")
    args = parser.parse_args()

    fraction = 3 if args.third_octave else 1
    bands = THIRD_OCTAVE_BANDS_HZ if args.third_octave else OCTAVE_BANDS_HZ

    if args.wav is not None:
        if not SOUNDFILE_AVAILABLE:
            raise ImportError(
                "soundfile is required to load WAV files. "
                "Install it with: pip install soundfile")
        rir, fs = sf.read(args.wav, always_2d=False)
        print(f"[INFO] Loaded '{args.wav}' | fs={fs} Hz | "
              f"length={len(rir)/fs:.3f} s")
    else:
        selectedRIR = select_wav_file()
        if selectedRIR:
            rir, fs = sf.read(selectedRIR, always_2d=False)
        else:
            print("No valid WAV file selected.")
            return

    # Filter bands to those below Nyquist
    nyq = fs / 2.0
    bands = [f for f in bands if f < nyq * 0.9]

    result = calculate_reverberation_time(
        rir=rir,
        fs=fs,
        bands=bands,
        fraction=fraction,
        method=args.method,
        plot=not args.no_plot,
    )

    return result



# run the code on a specified .wav file or on the synthetic RIR if no file is selecte
 

#Example of how to run this into your script
if __name__ == "__main__":

    main()  
