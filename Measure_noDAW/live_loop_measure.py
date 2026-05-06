import time
import sys
import threading

import numpy as np
import sounddevice as sd

from scipy.signal import fftconvolve

from lib import calibration


PLAYBACK_AUDIO_DEVICE = "Headphone"
RECORDING_AUDIO_DEVICE = "MacBook Pro Microphone"

SAMPLE_RATE = 48000                 # Hz    - sample rate for audio playback
BLOCK_LENGTH = 128                  # samples

F_START = 1                         # Hz    - sweep start frequency
F_FINAL = SAMPLE_RATE // 2          # Hz    - sweep end frequency (Nyquist)
T_SWEEP = 4                         # sec   - duration of the sine sweep
T_IDLE  = 2                         # sec   - silence appended after the sweep

VOLUME = 0.05 # chosen level to avoid clipping


def main() -> int:
    """Main function"""

    # Create sweep
    sweep = calibration.ess_gen_farina(
        F_START, F_FINAL, T_SWEEP, T_IDLE, SAMPLE_RATE,
        fade_in=128, cut_zerocross=True, sweep_gain=VOLUME
    )

    global_position = 0

    num_input_channels = 1  # To remain 1 for current code
    num_output_channels = 2 # Adjust as needed for your output device and auralization setup

    speaker_index = 1
    mic_index     = 0
    audio         = (sweep).astype(np.float32)  # 1-D, float32
    total_samples = len(audio)
    recorded   = np.zeros((total_samples, num_input_channels), dtype=np.float32)
    stop_event = threading.Event()

    print("Ready to sweep!")
    time.sleep(3)


    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time, _status) -> None:
        nonlocal global_position

        outdata.fill(0) # Clear the output buffer before writing new audio data - STANDARD FOR AUDIO CALLBACKS 
 
        pos = global_position
        remaining  = total_samples - pos
        chunk_size = min(frames, remaining)   # may be < frames on last block

        if chunk_size > 0:
            outdata[:chunk_size, speaker_index] = audio[pos : pos + chunk_size]
            recorded[pos : pos + chunk_size, mic_index] = indata[:chunk_size, mic_index]

        global_position = pos + chunk_size

        # Stop the stream once the entire sweep has been played and recorded.
        if global_position >= total_samples:
            stop_event.set()
            raise sd.CallbackStop

    try:
        with sd.Stream(
            device=(RECORDING_AUDIO_DEVICE, PLAYBACK_AUDIO_DEVICE),
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_LENGTH,
            channels=(num_input_channels, num_output_channels),
            callback=callback,
            dtype="float32",
       ):
            stop_event.wait()
    except sd.PortAudioError as error:
        print(f"Audio output error: {error}", file=sys.stderr)
        return 1


    mic_signal = recorded[:, mic_index]  # 1-D float32
    rir = calibration.ess_parse_farina(
        mic_signal, sweep, T_SWEEP, T_IDLE, SAMPLE_RATE, causality=False
    )

    # Test that the sweep * RIR approximates the recorded signal (for sanity check)
    recon_signal = fftconvolve(sweep, rir)[:len(mic_signal)]

    assert np.allclose(recon_signal, mic_signal, atol=1e-2, rtol=1e-2)

    tag = f"sp{speaker_index}"
    np.save(f"rir_{tag}.npy", rir)

    print("\nMeasurement complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
