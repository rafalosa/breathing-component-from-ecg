import numpy as np
from scipy.signal import welch
from scipy.interpolate import splev, splrep
from scipy.integrate import simps
from typing import Tuple
import matplotlib.pyplot as plt


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
        inside_range = np.r_[max_index - window_width_index // 2: max_index + window_width_index // 2 + 1]
        outside_range = np.r_[0:max_index - window_width_index // 2,
                              max_index + window_width_index // 2 + 1: EDR_spectrum.shape[0]]

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
    complement = zip(fractions, EDRS)
    sorted_complements = sorted(complement, reverse=False)

    sorted_fractions = np.array([el[0] for el in sorted_complements])
    sorted_EDR = np.array([el[1] for el in sorted_complements])

    return sorted_fractions, sorted_EDR


if __name__ == "__main__":

    pass
