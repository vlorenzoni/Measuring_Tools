"""
measurement_quality.py
======================
Quality assessment functions for sine-sweep RIR measurements.

Designed to slot into your measurement script after sd.playrec() and
swp.ess_parse_farina(), replacing the ad-hoc plots at the bottom.

Typical usage (drop-in replacement for your existing plots)
-----------------------------------------------------------
    from lib.measurement_quality import check_recording, check_rir, check_all

    # After sd.wait() and inside your mic loop:
    recording_quality = check_recording(
        recording[:, mic_idx],
        sweep=sweep,
        fs=SAMPLE_RATE,
        t_sweep=T_SWEEP,
        t_idle=T_IDLE,
        mic_name=mic_names[mic_idx],
        plot=True,
    )

    rir_quality = check_rir(
        rir,
        fs=SAMPLE_RATE,
        mic_name=mic_names[mic_idx],
        plot=True,
    )

    # Or run everything at once for all channels:
    results = check_all(recording, rirs, mic_names, sweep, SAMPLE_RATE, T_SWEEP, T_IDLE)

Functions
---------
    check_recording(rec, sweep, fs, t_sweep, t_idle, mic_name, plot)
        → checks the raw microphone recording (clipping, SNR, sweep presence)

    check_rir(rir, fs, mic_name, plot)
        → checks the deconvolved RIR (SNR, direct path, truncation, tail)

    check_all(recording, rirs, mic_names, sweep, fs, t_sweep, t_idle, plot)
        → runs both checks on every channel and prints a summary table
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ── thresholds (adjust to your setup) ────────────────────────────────────────

CLIP_THRESHOLD       = 0.99    # fraction of full scale — above = clipping
SNR_GOOD_DB          = 50      # minimum acceptable SNR (dB)
SNR_WARN_DB          = 35      # below this → warning
PEAK_GOOD_DBFS       = -6      # recording peak should be below this
PEAK_WARN_DBFS       = -20     # below this the recording may be too quiet
TRUNCATION_MARGIN_DB = 6       # if RIR end is this many dB above noise → truncated


# ── internal helpers ──────────────────────────────────────────────────────────

def _rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def _peak_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.max(np.abs(x)) + 1e-12)


def _noise_floor_db(x: np.ndarray, tail_fraction: float = 0.10) -> float:
    """RMS of the last *tail_fraction* of the signal — assumed to be silence."""
    tail = x[int((1 - tail_fraction) * len(x)):]
    return _rms_db(tail)


def _snr_db(x: np.ndarray, tail_fraction: float = 0.10) -> float:
    return _peak_db(x) - _noise_floor_db(x, tail_fraction)


def _flag(value, good, warn, higher_is_better=True):
    """Return ✅ / ⚠️ / ❌ emoji based on thresholds."""
    if higher_is_better:
        if value >= good:  return "✅"
        if value >= warn:  return "⚠️ "
        return "❌"
    else:
        if value <= good:  return "✅"
        if value <= warn:  return "⚠️ "
        return "❌"


def _smooth_env_db(x: np.ndarray, fs: int, win_ms: float = 10.0) -> np.ndarray:
    """Smoothed energy envelope in dB."""
    win = max(1, int(win_ms * 1e-3 * fs))
    env = np.convolve(x ** 2, np.ones(win) / win, mode='same')
    return 10 * np.log10(env + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Recording quality check
# ══════════════════════════════════════════════════════════════════════════════

def check_recording(
    rec:       np.ndarray,
    sweep:     np.ndarray,
    fs:        int,
    t_sweep:   float,
    t_idle:    float,
    mic_name:  str  = "",
    plot:      bool = True,
) -> dict:
    """
    Quality checks on the raw microphone recording.

    Checks
    ------
    - Clipping: any samples at or above CLIP_THRESHOLD × full scale
    - Peak level: is the signal loud enough but not too loud?
    - SNR: ratio of sweep peak to pre-sweep noise floor
    - Sweep presence: does the signal contain the expected sweep (via
      cross-correlation peak)? Detects missed trigger, wrong channel, etc.
    - DC offset: mean value of the signal

    Parameters
    ----------
    rec      : 1-D microphone recording (single channel)
    sweep    : the excitation sweep signal played during measurement
    fs       : sample rate in Hz
    t_sweep  : sweep duration in seconds
    t_idle   : silence appended after sweep, in seconds
    mic_name : label for the report (e.g. 'A')
    plot     : if True, show waveform + spectrogram + envelope

    Returns
    -------
    dict with keys: peak_db, noise_floor_db, snr_db, clipped,
                    dc_offset, sweep_found, sweep_lag_s, passed
    """
    label = f"Mic {mic_name}" if mic_name else "Recording"
    rec   = np.asarray(rec, dtype=float)
    n     = len(rec)

    # ── clipping ──────────────────────────────────────────────────────────────
    clip_count   = int(np.sum(np.abs(rec) >= CLIP_THRESHOLD))
    clipped      = clip_count > 0

    # ── levels ───────────────────────────────────────────────────────────────
    peak         = _peak_db(rec)
    # noise floor from a quiet window before the sweep starts
    pre_samples  = max(1, int(0.1 * fs))          # first 100ms
    noise_floor  = _rms_db(rec[:pre_samples])
    snr          = peak - noise_floor

    # ── DC offset ────────────────────────────────────────────────────────────
    dc           = float(np.mean(rec))
    dc_db        = 20 * np.log10(abs(dc) + 1e-12)

    # ── sweep presence via cross-correlation ─────────────────────────────────
    sweep_arr    = np.asarray(sweep, dtype=float)
    if sweep_arr.ndim > 1:
        sweep_arr = sweep_arr[:, 0]
    S            = len(sweep_arr)
    N_cc         = n + S - 1
    cc           = np.fft.irfft(
        np.fft.rfft(rec, n=N_cc) * np.conj(np.fft.rfft(sweep_arr, n=N_cc)),
        n=N_cc,
    )
    lag          = int(np.argmax(np.abs(cc)))
    sweep_lag_s  = lag / fs
    sweep_found  = 0 < sweep_lag_s < (n / fs - t_sweep)

    # ── console report ────────────────────────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  Recording quality — {label}")
    print(f"{'═'*54}")
    print(f"  Peak level    : {peak:>7.1f} dBFS  "
          f"{_flag(peak, PEAK_GOOD_DBFS, PEAK_WARN_DBFS, higher_is_better=False)}")
    print(f"  Noise floor   : {noise_floor:>7.1f} dBFS")
    print(f"  SNR           : {snr:>7.1f} dB    "
          f"{_flag(snr, SNR_GOOD_DB, SNR_WARN_DB)}")
    print(f"  Clipping      : {'YES ❌  (' + str(clip_count) + ' samples)' if clipped else 'none ✅'}")
    print(f"  DC offset     : {dc:>+.5f}  ({dc_db:.1f} dBFS)")
    print(f"  Sweep found   : {'YES ✅  at ' + f'{sweep_lag_s:.3f}s' if sweep_found else 'NOT FOUND ❌'}")

    passed = (not clipped) and (snr >= SNR_WARN_DB) and sweep_found
    print(f"  Overall       : {'✅ PASS' if passed else '⚠️  REVIEW NEEDED'}")
    print(f"{'═'*54}")

    if plot:
        _plot_recording(rec, sweep_arr, fs, label, lag, t_sweep, t_idle)

    return {
        "peak_db":       peak,
        "noise_floor_db": noise_floor,
        "snr_db":        snr,
        "clipped":       clipped,
        "clip_count":    clip_count,
        "dc_offset":     dc,
        "sweep_found":   sweep_found,
        "sweep_lag_s":   sweep_lag_s,
        "passed":        passed,
    }


def _plot_recording(rec, sweep, fs, label, lag, t_sweep, t_idle):
    t = np.arange(len(rec)) / fs
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"Recording quality — {label}", fontsize=12)

    # waveform
    ax = axes[0]
    ax.plot(t, rec, color="#2196F3", lw=0.5)
    ax.axhline( CLIP_THRESHOLD, color="r", ls="--", lw=0.8, label="Clip threshold")
    ax.axhline(-CLIP_THRESHOLD, color="r", ls="--", lw=0.8)
    if 0 < lag < len(rec):
        ax.axvspan(lag/fs, min((lag + int(t_sweep*fs))/fs, t[-1]),
                   alpha=0.15, color="orange", label="Sweep window")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # smoothed envelope in dB
    ax = axes[1]
    env_db = _smooth_env_db(rec, fs)
    ax.plot(t, env_db, color="#4CAF50", lw=0.8)
    ax.set_ylabel("Level (dBFS)")
    ax.set_title("Smoothed energy envelope")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(max(env_db.min(), -120), 5)
    ax.grid(True, alpha=0.3)

    # spectrogram
    ax = axes[2]
    ax.specgram(rec, Fs=fs, NFFT=1024, noverlap=512, cmap="inferno")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram")
    ax.set_xlim(0, t[-1])

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  2. RIR quality check
# ══════════════════════════════════════════════════════════════════════════════

def check_rir(
    rir:      np.ndarray,
    fs:       int,
    mic_name: str  = "",
    plot:     bool = True,
) -> dict:
    """
    Quality checks on a deconvolved Room Impulse Response.

    Checks
    ------
    - SNR: peak vs noise floor (last 10 % of file)
    - Direct sound: position and level of the main peak
    - Tail truncation: is the reverb still active at the end of the file?
    - Pre-ringing: significant energy before the direct sound (deconv artefact)
    - EDC shape: Schroeder curve and rough RT60 estimate

    Parameters
    ----------
    rir      : 1-D deconvolved room impulse response
    fs       : sample rate in Hz
    mic_name : label for the report
    plot     : if True, show waveform + EDC + estimated RT

    Returns
    -------
    dict with keys: peak_db, noise_floor_db, snr_db, direct_idx,
                    direct_time_ms, truncated, pre_ringing_db,
                    rt60_estimate_s, passed
    """
    label = f"Mic {mic_name}" if mic_name else "RIR"
    rir   = np.asarray(rir, dtype=float)
    n     = len(rir)

    # ── peak / direct sound ───────────────────────────────────────────────────
    direct_idx  = int(np.argmax(np.abs(rir)))
    peak        = _peak_db(rir)
    noise_floor = _noise_floor_db(rir, tail_fraction=0.10)
    snr         = peak - noise_floor

    # ── pre-ringing: energy in first half before direct sound ─────────────────
    pre_region  = rir[:max(1, direct_idx - int(0.001 * fs))]  # exclude 1ms before peak
    pre_ringing = _peak_db(pre_region) if len(pre_region) > 0 else -120.0
    pre_ringing_rel = pre_ringing - peak   # should be << 0 for a clean RIR

    # ── tail truncation ───────────────────────────────────────────────────────
    tail_level     = _rms_db(rir[int(0.90 * n):])
    end_margin     = tail_level - noise_floor
    truncated      = end_margin > TRUNCATION_MARGIN_DB

    # ── rough RT60 via Schroeder ───────────────────────────────────────────────
    rir_post       = rir[direct_idx:]
    energy         = rir_post ** 2
    schroeder      = np.cumsum(energy[::-1])[::-1]
    schroeder      = np.maximum(schroeder, 1e-20 * schroeder[0])
    edc            = 10 * np.log10(schroeder / schroeder[0])
    t_edc          = np.arange(len(edc)) / fs

    i5  = np.where(edc <= -5)[0]
    i25 = np.where(edc <= -25)[0]
    rt60 = (i25[0] - i5[0]) / fs * 3 if (len(i5) and len(i25)) else float('nan')

    # ── console report ────────────────────────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  RIR quality — {label}")
    print(f"{'═'*54}")
    print(f"  Peak level    : {peak:>7.1f} dBFS")
    print(f"  Noise floor   : {noise_floor:>7.1f} dBFS")
    print(f"  SNR           : {snr:>7.1f} dB    "
          f"{_flag(snr, SNR_GOOD_DB, SNR_WARN_DB)}")
    print(f"  Direct sound  : sample {direct_idx}  "
          f"({direct_idx/fs*1000:.1f} ms)")
    print(f"  Pre-ringing   : {pre_ringing_rel:>+.1f} dB rel. peak  "
          f"{'✅' if pre_ringing_rel < -40 else '⚠️  (check deconv)'}")
    print(f"  Tail truncated: {'YES ⚠️  (end margin ' + f'{end_margin:+.1f} dB)' if truncated else 'no ✅'}")
    if np.isfinite(rt60):
        print(f"  RT60 estimate : {rt60:.2f} s  (T20 extrapolation)")
    else:
        print(f"  RT60 estimate : N/A  (insufficient SNR for T20)")
    print(f"  RIR length    : {n/fs:.2f} s")

    passed = (snr >= SNR_WARN_DB) and (not truncated) and (pre_ringing_rel < -30)
    print(f"  Overall       : {'✅ PASS' if passed else '⚠️  REVIEW NEEDED'}")
    print(f"{'═'*54}")

    if plot:
        _plot_rir(rir, edc, t_edc, direct_idx, rt60, fs, label, noise_floor)

    return {
        "peak_db":          peak,
        "noise_floor_db":   noise_floor,
        "snr_db":           snr,
        "direct_idx":       direct_idx,
        "direct_time_ms":   direct_idx / fs * 1000,
        "truncated":        truncated,
        "pre_ringing_db":   pre_ringing_rel,
        "rt60_estimate_s":  rt60,
        "passed":           passed,
    }


def _plot_rir(rir, edc, t_edc, direct_idx, rt60, fs, label, noise_floor):
    n  = len(rir)
    t  = np.arange(n) / fs

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    fig.suptitle(f"RIR quality — {label}", fontsize=12)

    # waveform
    ax = axes[0]
    ax.plot(t, rir, color="#2196F3", lw=0.5)
    ax.axvline(direct_idx / fs, color="r", lw=1.2, ls="--", label="Direct sound")
    ax.set_ylabel("Amplitude")
    ax.set_title("Room Impulse Response")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # EDC
    ax = axes[1]
    ax.plot(t_edc, edc, color="#4CAF50", lw=1.2, label="EDC (Schroeder)")
    ax.axhline(noise_floor - 20 * np.log10(np.max(np.abs(rir)) + 1e-12),
               color="#888", lw=0.8, ls=":", label="Noise floor (approx)")
    ax.axhline(-60, color="r", lw=0.8, ls="--", label="−60 dB")
    if np.isfinite(rt60):
        ax.axvline(rt60, color="orange", lw=1.2, ls="--",
                   label=f"RT60 ≈ {rt60:.2f} s")
    ax.set_ylabel("Level (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Energy Decay Curve")
    ax.set_xlim(0, t_edc[-1])
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.show()


def _print_summary(results: list) -> None:
    """Print a compact summary table for a list of channel results."""
    n_ch = len(results)
    print(f"\n{'═'*74}")
    print(f"  MEASUREMENT SUMMARY — {n_ch} channel(s)")
    print(f"{'═'*74}")
    print(f"  {'Mic':>4}  {'Rec peak':>9}  {'Rec SNR':>8}  "
          f"{'Clip':>5}  {'RIR SNR':>8}  {'RT60':>7}  {'Trunc':>6}  {'Pass':>6}")
    print(f"  {'─'*66}")
    for r in results:
        rq    = r.get("recording") or {}
        iq    = r["rir"] or {}
        trunc = "YES" if iq.get("truncated") else "no"
        r_ok  = rq.get("passed", True)
        i_ok  = iq.get("passed", True)
        ok    = "✅" if (r_ok and i_ok) else "⚠️ "
        rt60  = iq.get("rt60_estimate_s", float("nan"))
        print(f"  {r['mic']:>4}  "
              f"{rq.get('peak_db', float('nan')):>9.1f}  "
              f"{rq.get('snr_db',  float('nan')):>8.1f}  "
              f"{'YES' if rq.get('clipped') else 'no':>5}  "
              f"{iq.get('snr_db',  float('nan')):>8.1f}  "
              f"{rt60:>7.2f}  "
              f"{trunc:>6}  "
              f"{ok:>6}")
    print(f"{'═'*74}\n")



# ══════════════════════════════════════════════════════════════════════════════
#  3. Combined check — all channels at once
# ══════════════════════════════════════════════════════════════════════════════

def check_all(
    recording:  np.ndarray,
    rirs:       list,
    mic_names:  list,
    sweep:      np.ndarray,
    fs:         int,
    t_sweep:    float,
    t_idle:     float,
    plot:       bool = True,
) -> list:
    """
    Run check_recording() and check_rir() on every channel and print a
    compact summary table at the end.

    Parameters
    ----------
    recording  : (n_samples, n_channels) array from sd.playrec()
    rirs       : list of 1-D RIR arrays, one per channel
    mic_names  : list of mic name strings, e.g. ['A','B','C','D','E','F']
    sweep      : excitation sweep signal
    fs         : sample rate in Hz
    t_sweep    : sweep duration in seconds
    t_idle     : idle silence in seconds
    plot       : if True, show plots for each channel

    Returns
    -------
    list of dicts, one per channel, each containing:
        'mic', 'recording', 'rir'  (sub-dicts from check_recording/check_rir)
    """
    results = []
    n_ch    = len(mic_names)

    for i, name in enumerate(mic_names):
        rec = recording[:, i] if recording.ndim > 1 else recording
        rir = rirs[i]

        rec_q = check_recording(rec, sweep, fs, t_sweep, t_idle,
                                mic_name=name, plot=plot)
        rir_q = check_rir(rir, fs, mic_name=name, plot=plot)
        results.append({"mic": name, "recording": rec_q, "rir": rir_q})

    _print_summary(results)
    return results


def check_folder(
    folder:    str,
    sweep:     np.ndarray,
    fs:        int,
    t_sweep:   float,
    t_idle:    float,
    rec_pattern:  str  = "mic_*.wav",
    rir_pattern:  str  = "rir_*.wav",
    plot:      bool = True,
) -> list:
    """
    Load all recordings and RIRs from a folder and run quality checks.

    Looks for files matching rec_pattern in <folder>/recordings/ and
    rir_pattern in <folder>/RIRs/, or falls back to searching folder
    directly if those subfolders don't exist.

    Alternatively pass the folder path directly and it will scan for
    any .wav files, pairing them by name.

    Simplest usage — just give the folder name:

        from lib.measurement_quality import check_folder
        check_folder("recordings/pos_test_probe", sweep, SAMPLE_RATE, T_SWEEP, T_IDLE)

    Parameters
    ----------
    folder      : path to the folder containing recordings and/or RIRs
    sweep       : excitation sweep signal
    fs          : sample rate in Hz
    t_sweep     : sweep duration in seconds
    t_idle      : idle silence in seconds
    rec_pattern : glob pattern for recording files (default "mic_*.wav")
    rir_pattern : glob pattern for RIR files      (default "rir_*.wav")
    plot        : if True, show plots per channel

    Returns
    -------
    list of result dicts (same format as check_all)
    """
    import soundfile as sf
    from pathlib import Path

    folder = Path(folder)

    # ── find recording and RIR files ─────────────────────────────────────────
    # try standard subfolder layout first
    rec_folder = folder / "recordings" if (folder / "recordings").exists() else folder
    rir_folder = folder / "RIRs"       if (folder / "RIRs").exists()       else folder

    rec_files = sorted(rec_folder.glob(rec_pattern))
    rir_files = sorted(rir_folder.glob(rir_pattern))

    # fallback: if no matches, just load all wav files from folder
    if not rec_files and not rir_files:
        all_wavs  = sorted(folder.glob("*.wav"))
        rec_files = [f for f in all_wavs if "rir" not in f.stem.lower()]
        rir_files = [f for f in all_wavs if "rir"     in f.stem.lower()]

    if not rec_files and not rir_files:
        print(f"No WAV files found in {folder}")
        return []

    print(f"Found {len(rec_files)} recording(s) and {len(rir_files)} RIR(s) in {folder}")

    results = []

    # ── process RIRs (always present) ────────────────────────────────────────
    for rir_path in rir_files:
        name = rir_path.stem   # use filename as label
        rir, rir_fs = sf.read(str(rir_path), dtype="float64")
        if rir.ndim > 1:
            rir = rir[:, 0]
        if rir_fs != fs:
            warnings.warn(f"{rir_path.name}: fs={rir_fs}, expected {fs}")

        rir_q = check_rir(rir, rir_fs, mic_name=name, plot=plot)

        # try to find a matching recording file (same mic letter/index)
        rec_q = None
        for rec_path in rec_files:
            if _stem_matches(rir_path.stem, rec_path.stem):
                rec, rec_fs = sf.read(str(rec_path), dtype="float64")
                if rec.ndim > 1:
                    rec = rec[:, 0]
                rec_q = check_recording(rec, sweep, rec_fs, t_sweep, t_idle,
                                        mic_name=name, plot=plot)
                break

        results.append({"mic": name, "recording": rec_q, "rir": rir_q})

    # ── process recordings with no matching RIR ───────────────────────────────
    rir_stems = {r.stem for r in rir_files}
    for rec_path in rec_files:
        if not any(_stem_matches(r, rec_path.stem) for r in rir_stems):
            name = rec_path.stem
            rec, rec_fs = sf.read(str(rec_path), dtype="float64")
            if rec.ndim > 1:
                rec = rec[:, 0]
            rec_q = check_recording(rec, sweep, rec_fs, t_sweep, t_idle,
                                    mic_name=name, plot=plot)
            results.append({"mic": name, "recording": rec_q, "rir": None})

    _print_summary(results)
    return results


def _stem_matches(rir_stem: str, rec_stem: str) -> bool:
    """Heuristic: two filenames 'match' if they share a common mic letter."""
    for ch in "ABCDEF":
        if ch in rir_stem.upper() and ch in rec_stem.upper():
            return True
    return False
