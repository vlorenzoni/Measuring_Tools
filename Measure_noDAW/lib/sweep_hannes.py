"""
    Module which contains functions relating to transfer-function measurements
    using Exponential Sine Sweep methods, as presented in [1] and [2].

    References
    ----------
    [1] A. Farina, “Advancements in impulse response measurements by sine
    sweeps,” Audio Engineering Society Convention 122, p. 22, 2007.
    [2] A. Novak, P. Lotton, and L. Simon, “Synchronized Swept-Sine: Theory,
    Application, and Implementation,” J. Audio Eng. Soc., vol. 63, no. 10,
    pp. 786–798, Nov. 2015

    ---
    Copyright 2024 Hannes Rosseel

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from typing import Tuple

import numpy as np
import scipy as sp


def ess_gen_farina(f_start: int, f_final: int, t_sweep: float, t_idle: float,
                   fs: int, fade_in: int = 0, cut_zerocross: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single Exponential Sine Sweep (ESS) and the inverse signal,
    which is used to calculate Room Impulse Responses (RIRs) according to
    Farina [1].

    Parameters
    ----------
        f_start: int
            Starting frequency in Hz.
        f_final: int
            Final frequency in Hz.
        t_sweep: float
            Duration of the sweep in seconds.
        fs: int
            Sampling frequency in Hz.
        fade_in: int, optional
            Number of window length fade in samples. Defaults to zero.
        cut_at_zerocross: bool, optional
            If this flag is set to `True`, cut ESS at the last zero-crossing,
            reducing the signal duration. This is done to prevent abrupt
            termination of the ESS (resulting in pulsive sound). Defaults to
            `False`.
    Returns
    -------
        sweep: np.ndarray
            Generated Exponential Sine Sweep.
        inverse: np.ndarray
            Inverse signal, which is the scaled time-inverse of the ESS.
    """
    R = float(f_final) / f_start
    C = (2 * f_final * np.log(R)) / ((f_final - f_start) * t_sweep)
    t = np.arange(0, int(fs * t_sweep)) / fs
    sweep = np.sin(((2 * math.pi * f_start * t_sweep) / math.log(R)) * (
                   np.power(R, (t / t_sweep)) - 1))

    if (fade_in > 0):
        sweep[0:fade_in] = (sweep[0:fade_in] * np.sin(np.arange(0, fade_in)
                                                      / fade_in * np.pi / 2))

    if (cut_zerocross):
        for idx, sample in enumerate(sweep[::-1]):
            if abs(sample) < 0.001:
                max_freq = (f_start * math.exp((t[-idx-1] / t_sweep) *
                                               math.log(R)))
                print("Warning: sweep cutoff at last zero-crossing. Final "
                      f"frequency is: {np.floor(max_freq)} Hz")
                sweep[-idx:] = np.zeros(idx)
                break

    inverse = C * np.power(R, -(t/t_sweep)) * np.flip(sweep)

    # Add idle time after ESS
    sweep = np.append(sweep, np.zeros(t_idle * fs))
    return (sweep, inverse)


def ess_parse_farina(sweep: np.ndarray, inverse: np.ndarray, t_sweep: float,
                     t_idle: float, fs: int, offset: int = 0, causality: bool = False
                     ) -> np.ndarray:
    """
    Process the input Exponential Sine Sweep (ESS) and output the
    resulting Room Impulse Response (RIR) according to Farina [1].

    Parameters
    ----------
        sweep: np.ndarray
            Input Exponential Sine Sweep (ESS).
        inverse: np.ndarray
            Inverse signal of the ESS.
        t_sweep: float
            Duration of the active sweep in seconds.
        t_idle: float
            Idle time in seconds following a single ESS.
        fs: int
            Sampling frequency in Hz
        offset: int, optional
            Offset in samples to compensate for acquisition system shifts. Value obtained from calibration with shortcuting speaker to mic in and recording sweep. Defaults to `0`.
        causality: bool, optional
            If this flag is set to `True`, only return the causal part of the
            RIR. Otherwise, return the full RIR. Defaults to `False`.
    Returns
    -------
        rir: np.ndarray
            The resulting Room Impulse Response.
    """
    if sweep.ndim > 1:
        raise Exception("Input has more than one dimension. Please input"
                        "a one dimensional vector containing the ESS.")
    duration = int(np.floor((t_sweep + t_idle) * fs))
    rir = np.array(sp.signal.fftconvolve(sweep[:duration], inverse,
                                         mode='full'))
    if causality:
        rir = rir[offset+int(t_sweep * fs - 1):duration+offset]
    return rir.real


def ess_gen_novak(f_start: int, f_final: int, t_sweep: float, t_idle: float,
                  fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Synchronized Exponential Sine Sweep (ESS) and its inverse-time
    signal, which is used to calculate Room Impulse Responses (RIRs) according
    to Novak et al. [2].

    Parameters
    ----------
        f_start: int
            Starting frequency in Hz.
        f_final: int
            Final frequency in Hz.
        t_sweep: float
            Estimated duration of the sweep in seconds.
        t_idle: float
            Idle time in seconds following a sweep. This idle time captures
            the remaining reverberation of the room.
        fs: int
            Sampling frequency in Hz.
    Returns
    -------
        sweep: np.ndarray
            Generated Exponential Sine Sweep.
        inverse_spec: np.ndarray
            Inverse filter of the generated sweep in the frequency domain.
    """

    # Generate sweep signal using eqs. (47) and (49), described in [2].
    L = round(f_start / np.log(f_final / f_start) * t_sweep) / f_start
    t_sweep = L * np.log(f_final / f_start)

    n_sweep = int(np.round(t_sweep * fs))
    n_total = n_sweep + t_idle * fs

    t = np.arange(0, int(np.ceil(t_sweep * fs))) / fs
    signal = np.zeros(n_total)

    # Calculate Synchronized ESS
    signal[:n_sweep] = np.sin(2 * np.pi * f_start * L * np.exp(t[:n_sweep]
                              / L))

    fft_length = int(2**np.ceil(np.log2(signal.shape[0])))
    f_axis = (fs / 2) * np.arange(0, fft_length) / fft_length

    # Ignore division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Inverse filter in frequency domain (single-sided spectrum) (eq. 43)
        inverse_spec = (2 * np.sqrt(f_axis / L) *
                        np.exp(-2j * np.pi * f_axis * L *
                        (1 - np.log(f_axis / f_start)) + 1j * np.pi / 4))
    inverse_spec[0] = 0  # Eliminate Inf DC component in spectrum
    return (signal, inverse_spec)


def ess_parse_novak(sweep: np.ndarray, inverse_spec: np.ndarray,
                    fs: int, t_idle: float = 0,
                    causality: bool = False) -> np.ndarray:
    """
    Calculated the Room Impulse Response (RIR) by convoluting the input
    Synchronized Exponential Sine Sweep (ESS) with the inverse signal,
    according to Novak et al. [2].

    Parameters
    ----------
        sweep: np.ndarray
            Input Exponential Sine Sweep (ESS).
        inverse_spec: np.ndarray
            Inverse filter of the generated sweep in the frequency domain.
        fs: int
            Sampling frequency in Hz
        t_idle: float, optional
            Idle time in seconds following a sweep. This idle time captures
            the remaining reverberation of the room. Defaults to `0`.
        causality: bool, optional
            If this flag is set to `True`, only return the causal part of the
            RIR. Otherwise, return the full RIR. Defaults to `False`.
    Returns
    -------
        rir: np.ndarray
            The resulting Room Impulse Response.
    """
    # ensure that the input sweep is 2D
    sweep = np.atleast_2d(sweep).T if len(sweep.shape) == 1 else sweep
    fft_length = int(2 ** np.ceil(np.log2(sweep.shape[0])))

    # Convert signal to FFT domain
    X = np.fft.rfft(sweep, n=(fft_length * 2 - 1), axis=0)
    pos_freq_spec = (X.T * inverse_spec).T
    pos_freq_spec[0] = 0  # Avoid infinity at DC
    h = np.fft.irfft(pos_freq_spec, n=fft_length, axis=0)

    if causality:
        return h[:int(t_idle * fs), :]
    else:
        return np.fft.ifftshift(h, axes=0)


def create_lin_perfect_seq(M: int, stretch_factor: int or None = None
                           ) -> np.ndarray:
    """
    Create a perfect linearly increasing sweep sequence of length M.
    The sweep is generated in the frequency domain and then transformed
    to the time domain using the IFFT.

    Currently, only even length sweeps are supported.

    Reference: C. Antweiler, A. Telle, P. Vary, and G. Enzner, “Perfect-sweep
               NLMS for time-variant acoustic system identification,” in 2012
               IEEE International Conference on Acoustics, Speech and Signal
               Processing (ICASSP), Kyoto, Japan: IEEE, Mar. 2012, pp. 517-520.
               doi: 10.1109/ICASSP.2012.6287930.

    Parameters
    ----------
    M : int
        Length of the perfect sweep sequence
    stretch_factor : int, optional
        The stretch factor determines the energy concentration of the sweep.
        - M / 2:  sweep covers the whole period and energy is equally
                  distributed over the whole period.
        - M / 4:  sweep starts at approximately M / 4, and ends at 3M/4,
                  spreading the energy of the sweep in only half a period.

    Returns
    -------
    p : ndarray of shape (M,)
        Perfect sweep sequence
    """
    assert M % 2 == 0, "Only even length perfect sweeps are supported."

    # Set stretch factor to half the length of the sweep
    if stretch_factor is None:
        stretch_factor = M // 2

    # initialization
    M_half = M // 2
    phase = np.zeros((M))

    k = np.arange(0, M_half)
    phase[:M_half] = (-4 * np.pi * stretch_factor * (k ** 2)) / (M ** 2)

    # Construct an odd-symetrical phase needed for a real valued signal
    # in time domain.
    phase[M_half] = 0
    phase[M_half + 1:] = -np.flip(phase[1:M_half])

    # Calculate complex representation with constant magnitude for IFFT
    spectrum = np.exp(1j * phase)

    # Calculate IFFT to gain sweep signal in time domain
    p = np.real_if_close(np.fft.ifft(spectrum, axis=-1))

    # Assert that p is real
    assert np.allclose(p.imag, 0), "Perfect sweep is not real valued."

    # Normalize signal to have a maximal amplitude of one
    p = p / np.max(np.abs(p))
    return p


def decorrelate_perfect_seq(measurement: np.ndarray, pseq: np.ndarray, v: int
                            ) -> np.ndarray:
    """
    Decorrelate a measured signal with a perfect sequence. The measured signal
    is decorrelated by calculating the cross correlation between the measured
    signal and the shifted perfect sequence. The cross correlation is
    normalized by the total energy of the perfect sequence.

    Parameters
    ----------
    measurement : ndarray of shape (num_samples, num_channels)
        Measured signal
    pseq : ndarray of shape (num_samples,)
        Perfect sequence
    v: index of measured signal to decorrelate
    Returns
    -------
    rir : ndarray of shape (num_samples, num_channels)
    """
    # Verify that pseq is a row vector
    assert pseq.ndim == 1, "Perfect Sequence must be a row vector"

    if measurement.ndim == 1:
        measurement = np.atleast_2d(measurement).T
    num_channels = measurement.shape[1]

    rirs = []
    pseq = np.roll(pseq, -v)

    pseq_fft = np.fft.fft(pseq)
    for channel in range(num_channels):
        # Calculate the circular cross correlation
        corr = np.fft.ifft(np.fft.fft(measurement[:, channel]) *
                           np.conj(pseq_fft)).real
        rirs.append(corr)

    rirs = np.array(rirs).T

    # Return normalized rirs
    return rirs / np.sum(pseq**2)


