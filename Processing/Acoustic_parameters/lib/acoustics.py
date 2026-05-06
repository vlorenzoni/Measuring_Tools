"""
    Module which contains functions relating to the calculation of Objective
    Room Acoustic parameters.

    References
    ----------
    [1] M. R. Schroeder, “New Method of Measuring Reverberation Time,” The
    Journal of the Acoustical Society of America, vol. 37, no. 3, pp. 409-412,
    Mar. 1965
    [2] A. Gade, “Acoustics in Halls for Speech and Music,” in Springer
    Handbook of Acoustics, Thomas D. Rossing, Ed. New York, NY: Springer New
    York, 2007, pp. 301-350.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def mag2db(signal: np.ndarray, input_mode: str = "amplitude") -> np.ndarray:
    """
    Convert magnitude to decibels.

    Parameters
    ----------
        signal: np.ndarray
            Input array, specified as scalar or vector.
        mode: {"amplitude", "power"}, optional
            Express input array as either `amplitude` or
            `power` measurement. Default input array is expressed as
            `amplitude`.
    Returns
    -------
        np.ndarray
            Magnitude measurement expressed in decibels.
    """
    scaling = 10 if input_mode == "power" else 20
    return scaling * np.log10(np.abs(signal))

def get_direct_path_idx(rir: np.ndarray, threshold: int = 3):
    abs_rir = abs(rir)
    direct_idx = abs_rir.argmax()

    # Set the threshold 3dB below the maximal value
    threshold = abs_rir[direct_idx] * 10 ** (-threshold / 20)
    peaks = find_peaks(abs_rir, height=threshold)[0]

    if len(peaks):
        direct_idx = peaks[0]

    return direct_idx


def energy_decay_curve(rir: np.ndarray, fs: int = 1, plot: bool = False) -> np.ndarray:
    """
    Calculate the energy decay curve from a causal input Room
    Impulse Response (RIR) using the Schroeder inverse-integration method [1].

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int, optional
            Sampling frequency of the input RIR in Hz. Defaults to `1`.
        plot: bool, optional
            If set to `True`, the energy decay curve will
            be plotted. Defaults to `False`.
    Returns
    -------
        edc: np.ndarray
            The energy decay curve in dB.
    """
    power = np.square(rir)
    decay_curve = np.cumsum(power[::-1])[::-1]
    decay_curve_db = mag2db(decay_curve, "power")
    decay_curve_db -= decay_curve_db[0]  # Normalize

    if plot:
        time_axis = np.arange(decay_curve_db.shape[0]) / fs
        plt.plot(time_axis, decay_curve_db)
    return decay_curve_db


def rt(
    rir: np.ndarray,
    fs: int,
    db: int = 60,
    db_start: int = 5,
    db_end: int = 35,
    plot: bool = False,
) -> float:
    """
    Calculate the Reverberation Time (T) from a causal input Room Impulse
    Response (RIR) using the Schroeder inverse-integration method [1]. Reverberation time
    is determined from the decay rate (dB/s), as found when fitting a
    regression line (determined from a relevant interval) to the Energy Decay
    Curve (EDC). The standard T is calculated by extrapolating the decay rate
    from -5 dB to -35 dB, which is denoted as $T_{30}$.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        db: int
            Reverberation time threshold.
        db_start: int, optional
            Sound pressure decay starting point for fitting a regression line
            to the EDC. Defaults to `5`.
        db_end: int, optional
            Sound pressure decay endpoint for fitting a regression line to
            the EDC. Defaults to `35`.
        plot: bool, optional
            If set to `True`, the RT calculations will be plotted. Defaults to
            `False`.
    Returns
    -------
        rt: float
            RT in seconds.
    """
    if db != 60:
        print("Warning: Reverberation time is usually calculated at 60 dB decay.")
    if abs(db_start) > abs(db_end):
        raise ValueError("Initial starting dB is larger than the final dB.")

    start = get_direct_path_idx(rir)  # Start at direct sound
    edc = energy_decay_curve(rir[(start + 1) :])
    t = np.arange(0, edc.shape[0]) / fs

    p1 = abs(edc + abs(db_start)).argmin()
    p2 = abs(edc + abs(db_end)).argmin()

    # Linear Ordinary Least Squares fit (y = mx + b)
    X = np.vstack([t[p1:p2], np.ones((p2 - p1))]).T
    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ edc[p1:p2])  # [m, b]
    rt = (-abs(db) - beta_hat[1]) / beta_hat[0]

    if plot:
        t_fit = (np.arange(0, np.round(rt * fs) + 5)) / fs
        X_fit = np.vstack([t_fit, np.ones(t_fit.shape)]).T
        y_fit = X_fit @ beta_hat

        ax = plt.gca()
        ax.plot(t, edc, color="k", label="Energy Decay Curve")
        ax.plot(t_fit, y_fit, color="g", linestyle="--", label="Linear fit")

        ax.axvline(
            t[p1],
            color="c",
            linestyle="--",
            label=f"-{abs(db_start)} dB reference point",
        )
        ax.axvline(
            t[p2], color="b", linestyle="--", label=f"-{abs(db_end)} dB reference point"
        )
        ax.axhline(-abs(db), color="r", linestyle="--", label=f"-{abs(db)} dB")
        ax.axvline(
            rt,
            color="r",
            linestyle="--",
            label="$T_{" + f"{abs(db_end - db_start)}" + "}$",
        )
        ax.legend()
    return rt


def edt(rir: np.ndarray, fs: int, plot: bool = False) -> float:
    """
    Calculate the Early Decay Time (EDT) of an input Room Impulse Response
    (RIR). The EDT is the time it takes for the RIR to decay to -60 dB. The
    decay rate is calculated using the interval from 0 dB to -10 dB, relative
    to the direct sound.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        plot: bool, optional
            If set to `True`, the EDT calculations will be plotted. Defaults to
            `False`.
    Returns
    -------
        edt: float
            The EDT in seconds.
    """
    return rt(rir, fs, 60, 0, 10, plot)


def energy_ratio(rir: np.ndarray, early: tuple, late: tuple) -> float:
    """
    Calculate the energy ratio between the early, and late part of the input
    Room Impulse Response (RIR).

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        early: Tuple(int, int)
            The start- and endpoint of the desired early reverberation
            in number of samples.
        late: Tuple(int, int)
            The start- and endpoint of the desired late reverberation
            in number of samples.
    Returns
    -------
        ratio: float
            Ratio between early and late energy.
    """
    power = np.square(rir)
    early_power = np.sum(power[early[0] : early[1]])
    late_power = np.sum(power[late[0] : late[1]])
    return early_power / late_power


def clarity(rir: np.ndarray, fs: int, threshold: float = 0.05) -> float:
    """
    Calculate the Clarity of an input Room Impulse Response (RIR). Clarity is
    the ratio between energy in the RIR before and after 50ms relative to the
    direct sound.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        threshold: float, optional
            The time threshold in seconds relative to the direct sound.
            Defaults to 50ms.
    Returns
    -------
        clarity: float
            Clarity expressed in dB.
    """
    direct_idx = get_direct_path_idx(rir)
    threshold_idx = int(direct_idx + threshold * fs)
    return 10 * np.log10(
        energy_ratio(rir, (direct_idx, threshold_idx), (threshold_idx, -1))
    )


def definition(rir: np.ndarray, fs: int, threshold: float = 0.050) -> float:
    """
    Calculate the Definition of an input Room Impulse Response (RIR).
    Definition describes the ratio between the early energy, and the
    total energy in a RIR.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        threshold: float, optional
            The time threshold in seconds relative to the direct sound.
            Defaults to 50ms.
        peak_thresh: float, optional
            Threshold to determine the direct sound peak. Defaults to 1.
    Returns
    -------
        definition: float
            Definition expressed as a ratio.
    """
    direct_idx = get_direct_path_idx(rir)
    threshold_idx = int(direct_idx + threshold * fs)
    return energy_ratio(rir, (direct_idx, threshold_idx), (direct_idx, -1))


def center_time(rir: np.ndarray, fs: int) -> float:
    """
    Calculate the center time t_s, which describes the center of gravity
    of the squared RIR. A low value corresponds to a clear sound, whereas
    higher values indicate dominance of the late, reverberant energy.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        peak_thresh: float, optional
            Threshold to determine the direct sound peak. Defaults to 1.
    Returns
    -------
        center_time: float
            Center time in seconds.
    """
    direct_idx = get_direct_path_idx(rir)
    t = np.arange(0, rir.shape[0] - direct_idx) / fs
    ct = np.sum(t * rir[direct_idx:] ** 2) / np.sum(rir[direct_idx:] ** 2)
    return ct.astype(float)


# Calculate the Direct to Reverberant Ratio (DRR)
def direct_to_reverberant(rir: np.ndarray, fs, correction: float = 0.0025) -> float:
    """
    Calculate the Direct to Reverberant Ratio (DRR) of an input Room Impulse
    Response (RIR). DRR is the ratio between the energy of the direct sound,
    and the energy of the reverberant sound.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        correction: float, optional
            Correction factor in seconds to account for the fact that the
            direct sound has a certain width. Defaults to 0.0025.
        peak_thresh: float, optional
            Threshold to determine the direct sound peak. Defaults to 1.
    Returns
    -------
        drr: float
            DRR expressed in dB.
    """
    direct_idx = get_direct_path_idx(rir, 15)

    start = max(int(direct_idx - correction * fs), 0)
    end = int(direct_idx + correction * fs)

    drr = 10 * np.log10(np.trapz(rir[start:end] ** 2) / np.trapz(rir[end + 1 :] ** 2))
    return drr
