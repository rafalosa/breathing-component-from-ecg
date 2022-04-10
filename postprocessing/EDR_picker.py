import numpy as np
from scipy.signal import welch
from scipy.interpolate import splev, splrep
from scipy.integrate import simps
from typing import Tuple
import wfdb.io
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# todo: Wrap it.

def sort_EDR(candidates: np.ndarray, time_domain: np.ndarray, window_width: float, sampling_frequency: float) \
        -> Tuple[np.ndarray, np.ndarray]:

    """
    :param candidates: Candidates for EDR, 2D np.ndarray with candidates in rows.
    :param time_domain: Timestamps of the R peaks extracted from the ECG signal.
    :param window_width: Frequency window for calculating the spectral power fraction. Given in Hz.
    :param sampling_frequency: Original ECG sampling frequency.
    :return: Returns cubic spline interpolated EDR candidates with corresponding fractions, sorted in ascending order
    with respect to the fractions, meaning - the best candidate for EDR signal is the 0-th element of the array.
    """

    EDRS = []
    fractions = []

    extended_domain = np.arange(time_domain[0], time_domain[-1], 1 / sampling_frequency)

    for EDR in candidates:

        # Interpolate candidate with cubic spline to restore original sampling frequency and regularity.
        EDR_spline = splrep(time_domain, EDR, s=0)
        interpolated_EDR = splev(extended_domain, EDR_spline, der=0)

        # Transform to frequency domain
        EDR_spectrum_domain, EDR_spectrum = welch(interpolated_EDR, sampling_frequency, nperseg=2**13, nfft=2**16)
        window_width_index = window_width / sampling_frequency * 2 * EDR_spectrum.shape[0]

        max_index = np.argmax(EDR_spectrum)  # Locate global maximum

        # Place window at global maximum with window_width

        left_bracket = int(max_index - window_width_index // 2)
        right_bracket = int(max_index + window_width_index // 2 + 1)

        inside_range = np.r_[left_bracket: right_bracket]
        outside_range = np.r_[0:left_bracket,
                              right_bracket: EDR_spectrum.shape[0]]

        EDR_spectrum_inside = EDR_spectrum[inside_range]
        EDR_spectrum_domain_inside = EDR_spectrum_domain[inside_range]

        EDR_spectrum_outside = EDR_spectrum[outside_range]
        EDR_spectrum_domain_outside = EDR_spectrum_domain[outside_range]

        # Calculate spectral power inside and outside the window
        spectral_power_inside = simps(EDR_spectrum_inside, EDR_spectrum_domain_inside)
        spectral_power_outside = simps(EDR_spectrum_outside, EDR_spectrum_domain_outside)

        # Calculate the outside/inside power fraction
        fractions.append(spectral_power_outside/spectral_power_inside)
        EDRS.append(interpolated_EDR)

    # Sort EDR candidates with ascending order with respect to the fractions
    zipped_results = zip(fractions, EDRS)
    sorted_results = sorted(zipped_results, reverse=False)

    sorted_fractions = np.array([el[0] for el in sorted_results])
    sorted_EDR = np.array([el[1] for el in sorted_results])

    return sorted_fractions, sorted_EDR


if __name__ == "__main__":

    ann = wfdb.io.rdann("../data/fantasia_wfdb/f1y05", "ecg")
    data = wfdb.io.rdrecord("../data/fantasia_wfdb/f1y05")

    respiration = data.p_signal[:,0]
    idxs = ann.sample

    pt_len = 100
    start = 700

    chosen = idxs[start:start + pt_len]
    chosen_time = np.array(chosen) * 1/250

    respiration_chosen_raw = np.array(respiration[chosen])

    fig = plt.figure(figsize=(16, 10))

    fig.add_subplot(3, 1, 1)
    plt.plot(chosen_time, respiration_chosen_raw)
    plt.title("Simulated EDR candidate")

    frac, edr = sort_EDR(np.array([respiration_chosen_raw]), chosen_time, .08, 250)

    fig.add_subplot(3, 1, 2)
    plt.plot(np.linspace(chosen_time[0], chosen_time[-1], edr[0].shape[0]), edr[0])
    plt.title("Interpolated EDR candidate")

    fig.add_subplot(3, 1, 3)
    respiration_filtered = savgol_filter(respiration[chosen[0]:chosen[-1]], 100, 1)
    plt.plot(np.linspace(chosen_time[0], chosen_time[-1], respiration_filtered.shape[0]), respiration_filtered)
    plt.title("Filtered reference")
    plt.xlabel("Time (s)")

    print(frac)

    plt.show()
