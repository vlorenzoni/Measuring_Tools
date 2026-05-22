import sounddevice as sd
import soundfile as sf
import numpy as np
import math
from pathlib import Path
import datetime
from lib import sweeps_methods as swp
import matplotlib.pyplot as plt
from lib.measurement_quality import check_folder

# ---------------------------------------------------------------------------
# ROOM & MEASUREMENT PARAMETERS
# ---------------------------------------------------------------------------
#   Typical values:
#     Anechoic / treated studio   : 0.1 – 0.3 s
#     Office / living room        : 0.3 – 0.8 s
#     Classroom / meeting room    : 0.8 – 1.5 s
#     Large hall / gymnasium      : 1.5 – 3.0 s
#     Church / cathedral          : 3.0 – 8.0 s
#

RT60_ESTIMATE = 1.0                 # sec - adjust before every session, sweep parameters depend on this.

#   T_SWEEP rule : when averaging, T_sweep >= 2 * RT60 is sufficient —
#                  SNR is split between sweep length and number of averages.
#                  (Single-sweep rule would be 6 * RT60, but that is wasteful here.)
#                  Floor at 3 s (minimum useful sweep length).
#
#   T_IDLE rule  : T_idle >= RT60 + 1 s (full tail capture + safety margin)
#                  floor at 3 s (minimum to avoid tail wrap-around in deconvolution)
#
#   Both are derived from the same RT60_ESTIMATE so they are always
#   internally consistent. Never set them independently.
#   math.ceil ensures integer seconds are always passed to ess_gen_farina.
#
#   Total measurement time = (N_AVERAGES + 1) * (T_SWEEP + T_IDLE)
#   e.g. RT60=1.0 s → T_SWEEP=3 s, T_IDLE=3 s, N=5 → 6 * 6 = 36 s

T_SWEEP = math.ceil(max(2.0 * RT60_ESTIMATE, 3.0))   # sec — ceil: always rounds up, safe for ess_gen_farina
T_IDLE  = math.ceil(max(RT60_ESTIMATE + 1.0, 3.0))   # sec — ceil: never shorter than intended

SAMPLE_RATE = 48000                 # Hz   - must match your audio interface
F_START     = 20                    # Hz   - practical lower limit for speakers
F_FINAL     = 22000                 # Hz   - slightly below Nyquist to avoid ADC filter ringing
VOLUME      = 0.89                  # gain - ~-1 dBFS, leaves headroom to avoid clipping

N_AVERAGES           = 5            # number of USEFUL repetitions to average.
                                    # Total sweeps played = N_AVERAGES + 1 (the +1 is a mandatory
                                    # latency-absorbing sweep that is always discarded).
                                    # So N_AVERAGES=5 → 6 sweeps played, 5 averaged.
                                    # SNR gain = 10*log10(N):
                                    #   4  →  +6.0 dB
                                    #   5  →  +7.0 dB
                                    #   8  →  +9.0 dB
                                    #   16 → +12.0 dB
                                    # Increase in noisy/non-stationary environments.
MIN_CLEAN            = 3            # minimum clean sweeps required after rejection —
                                    # raises RuntimeError if fewer survive, forcing you
                                    # to fix the noise source or increase N_AVERAGES.
OUTLIER_THRESHOLD_DB = 2.0          # dB  - global: repetitions whose total energy deviates
                                    #       more than this from the median are discarded.
                                    #       Tighter than 3 dB because noise during the sweep
                                    #       smears across the entire RIR after deconvolution.
                                    #       This is the primary defence against sweep contamination.
LOCAL_TRANSIENT_DB   = 10.0         # dB  - local: peak window energy above the local median,
                                    #       checked on the IDLE (silence) portion ONLY.
                                    #       The sweep portion cannot use a local median reference
                                    #       because the ESS spectral tilt means low-frequency
                                    #       windows always have far more energy than high-frequency
                                    #       ones — the local median is not flat and any threshold
                                    #       produces false positives on perfectly clean sweeps.
                                    #       Sweep contamination is caught by the tighter global check.
                                    #       Idle transients only corrupt the RIR tail; direct sound
                                    #       and early reflections are unaffected.
LOCAL_WINDOW_SEC     = 0.5          # sec - sliding window length for local transient check.
IDLE_NOISE_LEVEL     = 1e-5         # gain - noise injected into idle silence (-100 dBFS) to prevent
                                    #        DAC relay clicks on the RME Fireface (and similar interfaces)
                                    #        that activate a mute relay after ~1-2 s of silence.
                                    #        Level is well below the microphone noise floor and has
                                    #        negligible effect on the RIR noise floor after deconvolution.
                                    #        Set to 0.0 to disable if your interface does not have this issue.


# ENTER POSITION
position = "test_1mic_avg"    # Name of the position, used for folder naming when saving recordings and RIRs.

# SPECIFY MICROPHONE CHANNEL
mic_channel = 9  # Microphone channel number from soundcard or from totalmix.

# SPECIFY OFFSET
OFFSET = 9309                       # samples - measured integer from calibration, applied to correct
                                    # system latency. Only applied when causality=True. Use Calibration.py to measure.


base_folder = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Generate the sweep signal (single period)
# ---------------------------------------------------------------------------
sweep, inverse_sweep = swp.ess_gen_farina(
        F_START, F_FINAL, T_SWEEP, T_IDLE, SAMPLE_RATE,
        fade_in=128, cut_zerocross=True, sweep_gain=VOLUME
    )

period_len = len(sweep)                         # samples per sweep+idle period
# Derive sweep_len from period_len and T_IDLE rather than from T_SWEEP directly.
# ess_gen_farina with cut_zerocross=True may produce a sweep slightly shorter
# than T_SWEEP * SAMPLE_RATE, so computing from the other end is exact.
idle_len   = int(T_IDLE * SAMPLE_RATE)          # silence samples — T_IDLE is an integer, exact
sweep_len  = period_len - idle_len              # actual sweep samples as generated

# ---------------------------------------------------------------------------
# Playback: (N_AVERAGES + 1) repetitions in one continuous stream.
# Sweep 0 is discarded — it absorbs the audio interface I/O latency so that
# all N_AVERAGES useful slices are sample-accurately aligned to each other.
# Total sweeps played = N_AVERAGES + 1 = 6 (5 useful + 1 latency absorber).
# ---------------------------------------------------------------------------
# Build playback explicitly rather than np.tile so we can inject idle noise.
# Idle noise (-100 dBFS) prevents DAC relay clicks on interfaces (e.g. RME Fireface)
# that activate a mute/protection relay after ~1-2 s of silence.
n_periods = N_AVERAGES + 1
playback  = np.zeros(n_periods * period_len)
for _i in range(n_periods):
    playback[_i * period_len : (_i + 1) * period_len] = sweep

if IDLE_NOISE_LEVEL > 0:
    rng = np.random.default_rng(seed=0)   # fixed seed: reproducible, not correlated sweep-to-sweep
    for _i in range(n_periods):
        idle_start = _i * period_len + sweep_len
        idle_end   = (_i + 1) * period_len
        playback[idle_start:idle_end] += IDLE_NOISE_LEVEL * rng.standard_normal(idle_end - idle_start)

total_sec = n_periods * period_len / SAMPLE_RATE
print(f"Measurement parameters:")
print(f"  T_SWEEP={T_SWEEP} s, T_IDLE={T_IDLE} s, N_AVERAGES={N_AVERAGES}")
print(f"  Sweeps played: {N_AVERAGES + 1} (1 latency absorber + {N_AVERAGES} useful)")
print(f"  Total recording duration: {total_sec:.1f} s  ({total_sec/60:.1f} min)")

# ---------------------------------------------------------------------------
# Audio device setup
# ---------------------------------------------------------------------------
print("Available audio devices:")
print(sd.query_devices())

device_id_or_name = "Fireface UFX+"
print(f"Using device: {device_id_or_name}")

# ---------------------------------------------------------------------------
# Single continuous recording (never stop the stream between repetitions —
# this is what guarantees sample-accurate alignment across all slices)
# ---------------------------------------------------------------------------
recording_raw = sd.playrec(
    playback,
    samplerate=SAMPLE_RATE,
    input_mapping=mic_channel,
    output_mapping=[1],
    device=device_id_or_name
)
sd.wait()

recording_raw = recording_raw.flatten()

# ---------------------------------------------------------------------------
# Slice into N_AVERAGES periods, discarding sweep 0 (latency absorber).
# Each slice layout is exactly [sweep | idle] as expected by ess_parse_farina.
# ---------------------------------------------------------------------------
slices = []
for i in range(1, N_AVERAGES + 1):              # i=0 is the latency absorber, skip it
    start = i * period_len
    chunk = recording_raw[start : start + period_len]
    if len(chunk) < period_len:                 # guard: recording ended early (OS scheduling)
        print(f"  WARNING: sweep {i} is short ({len(chunk)} < {period_len} samples), discarding.")
        continue
    slices.append(chunk)

if len(slices) < MIN_CLEAN:
    raise RuntimeError(
        f"Only {len(slices)} complete slice(s) recorded — expected {N_AVERAGES}. "
        f"Recording ended early."
    )

# ---------------------------------------------------------------------------
# Two-stage outlier rejection
#   Stage 1 — global energy of the full period:
#              catches broadly contaminated sweeps (sustained HVAC, loud noise).
#              Threshold is tight (2 dB) because sweep contamination smears
#              across the entire RIR after deconvolution.
#
#   Stage 2 — local sliding-window on IDLE portion only:
#              catches short transients (door slam, footstep) that do not move
#              global energy enough for Stage 1. Idle-only because the ESS
#              spectral tilt makes local median an invalid reference on the
#              sweep portion — it always fires false positives on clean sweeps.
#              Idle transients only corrupt the RIR tail, not direct sound.
# ---------------------------------------------------------------------------
window_len = int(LOCAL_WINDOW_SEC * SAMPLE_RATE)
energies   = np.array([np.sum(s ** 2) for s in slices])
median_e   = np.median(energies)

# Print raw energies first so the user can diagnose threshold issues
# without having to add debug prints manually.
print("  Per-sweep energies (relative to median):")
_energies_tmp = np.array([np.sum(s ** 2) for s in slices])
_med_tmp      = np.median(_energies_tmp)
for _si, _e in enumerate(_energies_tmp):
    print(f"    Sweep {_si+1}: {10*np.log10(_e/_med_tmp):+.2f} dB  (threshold ±{OUTLIER_THRESHOLD_DB} dB)")

clean_slices  = []
rejection_log = []          # one entry per sweep, for the diagnostic plots

for i, (s, e) in enumerate(zip(slices, energies)):
    global_dev_db = 10 * np.log10(e / median_e)

    # Stage 1: global energy — primary defence for sweep contamination
    if abs(global_dev_db) >= OUTLIER_THRESHOLD_DB:
        rejection_log.append((i + 1, global_dev_db, "REJECTED – global energy"))
        continue

    # Stage 2: local transient in IDLE portion only
    idle_portion = s[sweep_len:]
    local_e = np.array([
        np.sum(idle_portion[j : j + window_len] ** 2)
        for j in range(0, len(idle_portion) - window_len, window_len // 2)
    ])
    if len(local_e) > 0:
        idle_peak_db = float((10 * np.log10(local_e / (np.median(local_e) + 1e-30) + 1e-30)).max())
        if idle_peak_db > LOCAL_TRANSIENT_DB:
            rejection_log.append((i + 1, global_dev_db,
                                  f"REJECTED – transient in IDLE ({idle_peak_db:.1f} dB) "
                                  f"— corrupts RIR tail only"))
            continue

    clean_slices.append(s)
    rejection_log.append((i + 1, global_dev_db, "OK"))

for sweep_n, dev, status in rejection_log:
    print(f"  Sweep {sweep_n}: {status}  (global ΔE = {dev:+.1f} dB)")

n_rejected = len(slices) - len(clean_slices)
if n_rejected > 0:
    print(f"  WARNING: {n_rejected}/{len(slices)} repetition(s) rejected as outliers.")
print(f"  Averaging {len(clean_slices)} clean repetitions.")

if len(clean_slices) < MIN_CLEAN:
    raise RuntimeError(
        f"Only {len(clean_slices)} clean sweep(s) survived rejection — "
        f"minimum required is {MIN_CLEAN}. "
        f"Fix the noise source or increase N_AVERAGES."
    )

# Average raw recordings, THEN deconvolve once — never average RIRs directly
averaged_recording = np.mean(clean_slices, axis=0)  # clean slices only — used for RIR

# ---------------------------------------------------------------------------
# Save the averaged raw recording
# ---------------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
recording_folder = base_folder / "Recordings" / f"pos_{position}"
recording_folder.mkdir(parents=True, exist_ok=True)
recording_file_path = recording_folder / f"recording_{timestamp}.wav"
print(recording_file_path)
sf.write(recording_file_path, averaged_recording, SAMPLE_RATE)

# ---------------------------------------------------------------------------
# Deconvolve once on the averaged signal
# ---------------------------------------------------------------------------
rir = swp.ess_parse_farina(
    averaged_recording, inverse_sweep, T_SWEEP, T_IDLE, SAMPLE_RATE,
    offset=OFFSET, causality=True
)

rir_folder = base_folder / "RIRs" / f"pos_{position}"
rir_folder.mkdir(parents=True, exist_ok=True)
rir_file_path = rir_folder / f"rir_{timestamp}.wav"
print(rir_file_path)
sf.write(rir_file_path, rir, SAMPLE_RATE)

# ---------------------------------------------------------------------------
# Diagnostics — Figure 1: per-repetition energy bar chart
# ---------------------------------------------------------------------------
bar_colors = [
    "red" if "REJECTED" in status else "steelblue"
    for _, _, status in rejection_log
]

plt.figure(figsize=(12, 4))
plt.bar(range(1, len(rejection_log) + 1), 10 * np.log10(energies / median_e), color=bar_colors)
plt.axhline(0, color="k", linewidth=0.8)
plt.axhline( OUTLIER_THRESHOLD_DB, color="r", linestyle="--", linewidth=0.8,
             label=f"±{OUTLIER_THRESHOLD_DB} dB global threshold")
plt.axhline(-OUTLIER_THRESHOLD_DB, color="r", linestyle="--", linewidth=0.8)
plt.title("Per-repetition energy deviation from median (red = rejected)")
plt.xlabel("Repetition #")
plt.ylabel("ΔEnergy (dB)")
plt.xticks(range(1, len(rejection_log) + 1))
plt.legend()
plt.grid()
plt.tight_layout()

# ---------------------------------------------------------------------------
# Diagnostics — Figure 2: per-repetition spectrograms
# Each sweep shown individually so noise events are visible before averaging
# hides them. Rejected sweeps have a red title, clean ones green.
# A vertical dashed line marks the sweep/idle boundary in each panel.
# ---------------------------------------------------------------------------
n_slices   = len(slices)
fig, axes  = plt.subplots(n_slices, 1, figsize=(12, 2 * n_slices), sharex=True)
if n_slices == 1:
    axes = [axes]   # ensure iterable when only one slice

sweep_boundary_sec = sweep_len / SAMPLE_RATE

for i, (s, (sweep_n, dev, status)) in enumerate(zip(slices, rejection_log)):
    axes[i].specgram(s, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
    axes[i].axvline(sweep_boundary_sec, color="white", linestyle="--",
                    linewidth=0.8, label="sweep/idle boundary")
    title_color = "red" if "REJECTED" in status else "green"
    axes[i].set_title(f"Sweep {sweep_n}: {status}  (ΔE = {dev:+.1f} dB)",
                      color=title_color, fontsize=9)
    axes[i].set_ylabel("Hz", fontsize=8)

axes[-1].set_xlabel("Time [sec]")
plt.suptitle("Per-repetition spectrograms  |  red title = rejected", fontsize=11, y=1.002)
plt.tight_layout()

# ---------------------------------------------------------------------------
# Diagnostics — Figure 3: estimated RIR
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(rir)
plt.title("Estimated RIR (averaged)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(rir))
plt.grid()
plt.tight_layout()

plt.show()