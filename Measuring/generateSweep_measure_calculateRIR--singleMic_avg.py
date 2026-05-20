import sounddevice as sd
import soundfile as sf
import numpy as np
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
#
#   Total measurement time = (N_AVERAGES + 1) * (T_SWEEP + T_IDLE)
#   e.g. RT60=1.0 s → T_SWEEP=3 s, T_IDLE=3 s, N=6 → 7 * 6 = 42 s

T_SWEEP = max(2.0 * RT60_ESTIMATE, 3.0)   # sec  — shorter OK because averaging compensates for SNR
T_IDLE  = max(RT60_ESTIMATE + 1.0, 3.0)   # sec  — silence after sweep: must be > RT60. Here fixed to RT60 + 1 sec.

SAMPLE_RATE = 48000                 # Hz   - must match your audio interface
F_START     = 20                    # Hz   - practical lower limit for speakers
F_FINAL     = 22000                 # Hz   - slightly below Nyquist to avoid ADC filter ringing
VOLUME      = 0.89                  # gain - ~-1 dBFS, leaves headroom to avoid clipping

N_AVERAGES           = 6            # number of useful repetitions to average.
                                    # SNR gain = 10*log10(N):
                                    #   4  →  +6.0 dB
                                    #   6  →  +7.8 dB
                                    #   8  →  +9.0 dB
                                    #   16 → +12.0 dB
                                    # Increase in noisy/non-stationary environments.
MIN_CLEAN            = 3            # minimum clean sweeps required after rejection —
                                    # raises RuntimeError if fewer survive, forcing you
                                    # to fix the noise source or increase N_AVERAGES.
OUTLIER_THRESHOLD_DB = 3.0          # dB  - global: repetitions whose total energy deviates
                                    #       more than this from the median are discarded.
                                    #       Catches broadly noisy sweeps (sustained HVAC burst etc).
LOCAL_TRANSIENT_DB   = 10.0         # dB  - local: peak window energy in the idle (silence) portion
                                    #       above the noise floor of that same idle section.
                                    #       Catches short transients (door slam, footstep) that do
                                    #       not move global energy enough for stage 1 to catch.
                                    #       Checked on idle only — sweep portion has natural spectral
                                    #       energy variation that would trigger false positives.
LOCAL_WINDOW_SEC     = 0.5          # sec - sliding window length for local transient check.


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

period_len  = len(sweep)                        # samples per sweep+idle period
sweep_len   = int(T_SWEEP * SAMPLE_RATE)        # samples of active sweep only
idle_len    = period_len - sweep_len            # samples of silence only

# Build the full playback signal: (N_AVERAGES + 1) repetitions.
# The +1 discarded first sweep absorbs the I/O latency of the audio interface,
# guaranteeing all N_AVERAGES useful slices are sample-accurately aligned.
playback = np.tile(sweep, N_AVERAGES + 1)

total_sec = (N_AVERAGES + 1) * period_len / SAMPLE_RATE
print(f"Measurement parameters:")
print(f"  T_SWEEP={T_SWEEP:.1f} s, T_IDLE={T_IDLE:.1f} s, N_AVERAGES={N_AVERAGES}")
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
# Slice into individual repetitions, discard first (latency absorption),
# reject outliers, average in the raw domain BEFORE deconvolution
# ---------------------------------------------------------------------------
slices = []
for i in range(1, N_AVERAGES + 1):              # skip i=0 (latency sweep)
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
#   Stage 1 — global energy of the full period  : catches broadly contaminated
#              sweeps (sustained HVAC burst, loud background noise)
#   Stage 2 — local sliding-window on the IDLE  : catches short transients
#              (door slam, footstep) that do not move global energy enough
#              for stage 1. Checked on idle only — the sweep portion has
#              natural spectral energy variation that causes false positives.
# ---------------------------------------------------------------------------
window_len = int(LOCAL_WINDOW_SEC * SAMPLE_RATE)
energies   = np.array([np.sum(s ** 2) for s in slices])
median_e   = np.median(energies)

clean_slices  = []
rejection_log = []          # one entry per sweep, for the diagnostic plot

for i, (s, e) in enumerate(zip(slices, energies)):
    global_dev_db = 10 * np.log10(e / median_e)

    # Stage 1: global energy
    if abs(global_dev_db) >= OUTLIER_THRESHOLD_DB:
        rejection_log.append((i + 1, global_dev_db, "REJECTED – global energy"))
        continue

    # Stage 2: sliding-window local transient check on idle portion only
    idle_portion = s[sweep_len:]                # silence section only
    local_e = np.array([
        np.sum(idle_portion[j : j + window_len] ** 2)
        for j in range(0, len(idle_portion) - window_len, window_len // 2)
    ])
    if len(local_e) > 0:
        noise_floor  = np.median(local_e)
        local_peak_db = 10 * np.log10((local_e.max() / (noise_floor + 1e-30)) + 1e-30)
        if local_peak_db > LOCAL_TRANSIENT_DB:
            rejection_log.append((i + 1, global_dev_db,
                                  f"REJECTED – local transient {local_peak_db:.1f} dB in idle"))
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
averaged_recording = np.mean(clean_slices, axis=0)

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
# Diagnostics: show the averaged recording + per-slice energies
# ---------------------------------------------------------------------------
bar_colors = [
    "red" if "REJECTED" in status else "steelblue"
    for _, _, status in rejection_log
]

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.bar(range(1, len(rejection_log) + 1), 10 * np.log10(energies / median_e), color=bar_colors)
plt.axhline(0, color="k", linewidth=0.8)
plt.axhline( OUTLIER_THRESHOLD_DB, color="r", linestyle="--", linewidth=0.8, label=f"±{OUTLIER_THRESHOLD_DB} dB global threshold")
plt.axhline(-OUTLIER_THRESHOLD_DB, color="r", linestyle="--", linewidth=0.8)
plt.title("Per-repetition energy deviation from median (red = rejected)")
plt.xlabel("Repetition #")
plt.ylabel("ΔEnergy (dB)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(averaged_recording)
plt.title(f"Averaged Recording ({len(clean_slices)} repetitions)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(averaged_recording))
plt.grid()

plt.subplot(3, 1, 3)
plt.specgram(averaged_recording, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
plt.title("Spectrogram of Averaged Recording")
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.tight_layout()

# ---------------------------------------------------------------------------
# RIR plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(rir)
plt.title("Estimated RIR (averaged)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.xlim(0, len(rir))
plt.grid()
plt.show()