"""
acoustics.py
===================
Objective room acoustic parameter estimation from Room Impulse Responses (RIRs).

ISO 3382-1 compliant. Import and call functions directly from other scripts.

Example usage
─────────────
    from lib.acoustics import reverberation_time

    rir, fs = sf.read("my_rir.wav")
    if rir.ndim > 1:
        rir = rir[:, 0]

    result = reverberation_time(rir, fs, method="T30", plot=True)
    # → prints per-band RT table
    # → shows broadband EDC plot
    # → shows per-band EDC plot (one subplot per octave band)

    # Extra ISO 3382 parameters
    from lib.acoustics import edt, clarity, definition, center_time, direct_to_reverberant

References
──────────
    [1] M. R. Schroeder, "New Method of Measuring Reverberation Time,"
        JASA, vol. 37, no. 3, pp. 409-412, 1965.
    [2] A. Gade, "Acoustics in Halls for Speech and Music," in Springer
        Handbook of Acoustics, T. D. Rossing, Ed., 2007, pp. 301-350.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfilt


# ── ISO 266 band definitions ──────────────────────────────────────────────────

OCTAVE_BANDS_HZ = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

THIRD_OCTAVE_BANDS_HZ = [
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000,
]

_METHOD_WINDOWS = {
    "T20": (5.0, 25.0),
    "T30": (5.0, 35.0),
    "T60": (5.0, 65.0),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Internal building blocks
# ══════════════════════════════════════════════════════════════════════════════

def mag2db(signal: np.ndarray, input_mode: str = "amplitude") -> np.ndarray:
    scaling = 10 if input_mode == "power" else 20
    return scaling * np.log10(np.abs(signal) + 1e-300)


def get_direct_path_idx(rir: np.ndarray, threshold_db: float = 3.0) -> int:
    """
    Return the sample index of the direct sound using find_peaks.
    More robust than argmax: returns the first qualifying peak rather than
    the absolute maximum, which may be a strong early reflection.
    """
    abs_rir    = np.abs(rir)
    peak_amp   = abs_rir.max()
    min_height = peak_amp * 10 ** (-threshold_db / 20.0)
    peaks, _   = find_peaks(abs_rir, height=min_height)
    return int(peaks[0]) if len(peaks) else int(abs_rir.argmax())


def energy_decay_curve(rir: np.ndarray) -> np.ndarray:
    """
    Schroeder backward integration → normalised EDC in dB.
    Zero-guard prevents log(0) on silent tails.
    """
    energy    = rir ** 2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder = np.maximum(schroeder, 1e-20 * schroeder[0])
    return 10.0 * np.log10(schroeder / schroeder[0])


def bandpass_filter(signal: np.ndarray, fc: float, fs: float,
                    order: int = 6, fraction: int = 1) -> np.ndarray:
    """Octave (fraction=1) or 1/3-octave (fraction=3) Butterworth bandpass."""
    factor = 2.0 ** (1.0 / (2 * fraction))
    fl, fh = fc / factor, fc * factor
    nyq    = fs / 2.0
    if fl <= 0 or fh >= nyq:
        return np.zeros_like(signal)
    sos = butter(order, [fl / nyq, fh / nyq], btype="bandpass", output="sos")
    return sosfilt(sos, signal)


def _fit_edc(edc: np.ndarray, fs: float,
             db_start: float, db_end: float) -> tuple:
    """OLS regression on EDC window → (rt_sec, r_value, fit_line)."""
    n = len(edc)
    t = np.arange(n) / fs

    i_upper = int(np.argmin(np.abs(edc + db_start)))
    i_lower = int(np.argmin(np.abs(edc + db_end)))

    if i_lower <= i_upper:
        return np.nan, np.nan, np.full(n, np.nan)

    seg_t, seg_edc   = t[i_upper:i_lower], edc[i_upper:i_lower]
    slope, intercept = np.polyfit(seg_t, seg_edc, 1)

    if slope >= 0:
        return np.nan, np.nan, np.full(n, np.nan)

    rt_sec   = (-60.0 - intercept) / slope
    seg_fit  = slope * seg_t + intercept
    ss_res   = np.sum((seg_edc - seg_fit) ** 2)
    ss_tot   = np.sum((seg_edc - np.mean(seg_edc)) ** 2)
    r_value  = -np.sqrt(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-30)))
    fit_line = slope * t + intercept
    return rt_sec, r_value, fit_line


# ══════════════════════════════════════════════════════════════════════════════
#  Main public function
# ══════════════════════════════════════════════════════════════════════════════

def rt(
    rir:      np.ndarray,
    fs:       float,
    method:   str  = "T30",
    bands:    list = None,
    fraction: int  = 1,
    plot:     bool = True,
) -> dict:
    """
    Calculate reverberation time from a single-channel RIR.

    Computes and (optionally) plots:
      1. Broadband EDC with regression line — one figure
      2. Per-band EDC with regression lines — one figure, one subplot per band

    Parameters
    ----------
    rir      : 1-D room impulse response (multi-channel: first channel used)
    fs       : sample rate in Hz
    method   : 'T20', 'T30' (default), or 'T60'
    bands    : centre frequencies in Hz. Default: ISO octave bands 63–8000 Hz
    fraction : 1 = octave bands (default), 3 = 1/3-octave bands
    plot     : if True, show both broadband and per-band plots

    Returns
    -------
    dict
        'broadband_rt'  – broadband RT in seconds (extrapolated to −60 dB)
        'bands_hz'      – centre frequencies used
        'rt_seconds'    – RT per band (NaN if estimation failed)
        'r_values'      – Pearson r per band (−1 = perfect linear decay)
        'rt_mean'       – mean RT across all valid bands
        'rt_mid'        – ISO 3382 mid-freq average (500 Hz + 1 kHz)
    """
    method = method.upper()
    if method not in _METHOD_WINDOWS:
        raise ValueError(f"method must be one of {list(_METHOD_WINDOWS.keys())}")

    db_start, db_end = _METHOD_WINDOWS[method]

    if bands is None:
        bands = OCTAVE_BANDS_HZ

    rir = np.asarray(rir, dtype=float)
    if rir.ndim > 1:
        rir = rir[:, 0]

    # ── align to direct sound once ────────────────────────────────────────────
    start       = get_direct_path_idx(rir)
    rir_aligned = rir[start + 1:]

    # ── broadband ─────────────────────────────────────────────────────────────
    edc_broad              = energy_decay_curve(rir_aligned)
    t_broad                = np.arange(len(edc_broad)) / fs
    rt_broad, _, fit_broad = _fit_edc(edc_broad, fs, db_start, db_end)

    # ── per band ──────────────────────────────────────────────────────────────
    rt_list, r_list, edc_list, fit_list = [], [], [], []

    for fc in bands:
        filtered       = bandpass_filter(rir_aligned, fc, fs, fraction=fraction)
        edc            = energy_decay_curve(filtered)
        rv, r, fit     = _fit_edc(edc, fs, db_start, db_end)
        rt_list.append(rv)
        r_list.append(r)
        edc_list.append(edc)
        fit_list.append(fit)

    rt_array = np.array(rt_list, dtype=float)
    r_array  = np.array(r_list,  dtype=float)
    rt_mean  = float(np.nanmean(rt_array)) if np.any(np.isfinite(rt_array)) else np.nan
    mid_idx  = [i for i, f in enumerate(bands) if f in (500, 1000)]
    rt_mid   = float(np.nanmean(rt_array[mid_idx])) if mid_idx else np.nan

    # ── console report ────────────────────────────────────────────────────────
    label = "Octave" if fraction == 1 else "1/3-Octave"
    print("=" * 60)
    print(f"  Reverberation Time ({method}) — {label} Bands")
    print("=" * 60)
    print(f"  {'Broadband':>16}   "
          f"{'N/A' if not np.isfinite(rt_broad) else f'{rt_broad:.3f}':>8}")
    print(f"  {'Centre Freq (Hz)':>16}   {'RT (s)':>8}   {'r':>7}")
    print("  " + "─" * 42)
    for fc, rv, r in zip(bands, rt_array, r_array):
        rt_str = f"{rv:.3f}" if np.isfinite(rv) else "   N/A"
        r_str  = f"{r:.4f}"  if np.isfinite(r)  else "   N/A"
        print(f"  {fc:>16}   {rt_str:>8}   {r_str:>7}")
    print("  " + "─" * 42)
    print(f"  {'Mean RT (all bands)':>22}   {rt_mean:.3f} s")
    print(f"  {'RT_mid (500 + 1 kHz)':>22}   {rt_mid:.3f} s")
    print("=" * 60)

    # ── plots ─────────────────────────────────────────────────────────────────
    if plot:
        # Figure 1 – broadband
        fig, ax = plt.subplots(num=1, figsize=(10, 4))
        ax.plot(t_broad, edc_broad, color="k", label="EDC (broadband)")
        if np.isfinite(rt_broad):
            p1 = int(np.argmin(np.abs(edc_broad + db_start)))
            p2 = int(np.argmin(np.abs(edc_broad + db_end)))
            n_fit = min(int(np.round(rt_broad * fs)) + 5, len(t_broad))
            ax.plot(t_broad[:n_fit], fit_broad[:n_fit], "g--", label="Linear fit")
            ax.axvline(t_broad[p1], color="c", linestyle="--",
                       label=f"−{db_start:.0f} dB")
            ax.axvline(t_broad[p2], color="b", linestyle="--",
                       label=f"−{db_end:.0f} dB")
            ax.axhline(-60, color="r", linestyle="--", label="−60 dB")
            ax.axvline(rt_broad, color="r", linestyle="--",
                       label=f"{method} = {rt_broad:.2f} s")
        ax.set_title(f"Energy Decay Curve — Broadband {method}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Level (dB)")
        ax.set_ylim(-80, 5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
    

        # Figure 2 – per band
        n_bands = len(bands)
        ncols   = 4
        nrows   = int(np.ceil(n_bands / ncols))
        fig2, axes = plt.subplots(nrows, ncols, num=2, figsize=(16, nrows * 3.5),
                                   constrained_layout=True)
        axes = np.array(axes).flatten()
        bl   = "Oct" if fraction == 1 else "1/3-Oct"

        for idx, (fc, edc, fit, rv) in enumerate(
                zip(bands, edc_list, fit_list, rt_array)):
            ax  = axes[idx]
            t   = np.arange(len(edc)) / fs
            ax.plot(t, edc, color="#2196F3", lw=1.2, label="EDC")
            if np.isfinite(rv):
                valid = np.isfinite(fit)
                ax.plot(t[valid], fit[valid], color="#F44336", lw=1.5, ls="--",
                        label=f"{method} = {rv:.2f} s")
            ax.axhline(-60, color="#888", lw=0.8, ls=":", label="−60 dB")
            ax.set_xlim(0, t[-1])
            ax.set_ylim(-80, 5)
            ax.set_title(f"{fc} Hz ({bl})", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Level (dB)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            

        for ax in axes[n_bands:]:
            ax.set_visible(False)

        fig2.suptitle(f"Energy Decay Curves per Band — {method}", fontsize=13)
        plt.show()


    return {
        "broadband_rt": rt_broad,
        "bands_hz":     bands,
        "rt_seconds":   rt_array.tolist(),
        "r_values":     r_array.tolist(),
        "rt_mean":      rt_mean,
        "rt_mid":       rt_mid,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Additional ISO 3382 parameters
# ══════════════════════════════════════════════════════════════════════════════

def edt(rir: np.ndarray, fs: int, plot: bool = False) -> float:
    """Early Decay Time (0 to −10 dB window, extrapolated to 60 dB)."""
    rir   = np.asarray(rir, dtype=float)
    if rir.ndim > 1: rir = rir[:, 0]
    start = get_direct_path_idx(rir)
    edc   = energy_decay_curve(rir[start + 1:])
    t     = np.arange(len(edc)) / fs
    rv, _, fit = _fit_edc(edc, fs, 0, 10)
    if plot:
        fig, ax = plt.subplots(num=3, figsize=(10, 4))
        ax.plot(t, edc, color="k", label="EDC")
        if np.isfinite(rv):
            n_fit = min(int(np.round(rv * fs)) + 5, len(t))
            ax.plot(t[:n_fit], fit[:n_fit], "g--", label=f"EDT = {rv:.2f} s")
        ax.set_title("Early Decay Time")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Level (dB)")
        ax.set_ylim(-80, 5); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.show()
    return rv


def _energy_ratio(rir: np.ndarray, early: tuple, late: tuple) -> float:
    p = rir ** 2
    return np.sum(p[early[0]:early[1]]) / np.sum(p[late[0]:late[1]])


def clarity(rir: np.ndarray, fs: int, threshold: float = 0.050) -> float:
    """C50 (threshold=0.050 s) or C80 (threshold=0.080 s) in dB."""
    rir = np.asarray(rir, dtype=float)
    if rir.ndim > 1: rir = rir[:, 0]
    direct_idx    = get_direct_path_idx(rir)
    threshold_idx = int(direct_idx + threshold * fs)
    return 10.0 * np.log10(
        _energy_ratio(rir, (direct_idx, threshold_idx), (threshold_idx, len(rir)))
    )


def definition(rir: np.ndarray, fs: int, threshold: float = 0.050) -> float:
    """D50 (threshold=0.050 s) or D80 (threshold=0.080 s) as a ratio 0–1."""
    rir = np.asarray(rir, dtype=float)
    if rir.ndim > 1: rir = rir[:, 0]
    direct_idx    = get_direct_path_idx(rir)
    threshold_idx = int(direct_idx + threshold * fs)
    return _energy_ratio(rir, (direct_idx, threshold_idx), (direct_idx, len(rir)))


def center_time(rir: np.ndarray, fs: int) -> float:
    """Centre time Ts in seconds."""
    rir = np.asarray(rir, dtype=float)
    if rir.ndim > 1: rir = rir[:, 0]
    direct_idx = get_direct_path_idx(rir)
    tail       = rir[direct_idx:]
    t          = np.arange(len(tail)) / fs
    return float(np.sum(t * tail ** 2) / np.sum(tail ** 2))


def direct_to_reverberant(rir: np.ndarray, fs: int,
                           correction: float = 0.0025) -> float:
    """Direct-to-Reverberant Ratio in dB."""
    rir = np.asarray(rir, dtype=float)
    if rir.ndim > 1: rir = rir[:, 0]
    direct_idx = get_direct_path_idx(rir, threshold_db=15)
    start      = max(int(direct_idx - correction * fs), 0)
    end        = int(direct_idx + correction * fs)
    return float(10.0 * np.log10(
        np.trapezoid(rir[start:end] ** 2) /
        np.trapezoid(rir[end + 1:] ** 2)
    ))
