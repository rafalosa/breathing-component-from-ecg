import numpy as np
from scipy.signal import welch
from scipy.interpolate import splev, splrep
from scipy.integrate import simps
from typing import Tuple, List
import wfdb.io
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class EDRPicker:

    __slots__ = {"_spectral_power_window",
                 "_sampling_frequency",
                 "_spline_smoothing",
                 "_spline_derivative",
                 "_time_boundaries",
                 "_samples_per_segment",
                 "_nfft",
                 "candidates_raw",
                 "_power_spectra"}

    def __init__(self, time_boundaries: Tuple[float, float]):
        self._time_boundaries = time_boundaries
        self._spline_smoothing = 0
        self._spline_derivative = 0
        self._sampling_frequency = 100
        self._spectral_power_window = .1
        self._samples_per_segment = 2 ** 10
        self._nfft = 2 ** 10
        self._power_spectra: List[Tuple[np.ndarray, np.ndarray]] = []

    def set_spline_params(self, smoothing: float = 0, derivative: int = 0, sampling_frequency: float = 100) -> None:
        """
        Set spline interpolation/approximation parameters.
        :param smoothing: Smoothing parameter of the fitted spline.
        :param derivative: Derivative order of the spline.
        :param sampling_frequency: Desired sampling frequency in Hz of the fitted signal.
        :return: None.
        """
        self._spline_smoothing = smoothing
        self._spline_derivative = derivative
        self._sampling_frequency = sampling_frequency

    def set_spectral_params(self, window_width: float = .1, samples_per_segment: int = 2 ** 10,
                            nfft_: int = 2 ** 10) -> None:
        """
        Set spectral analysis parameters.
        :param window_width: Window width in Hz around the global maximum of the spectrum for each EDR candidate.
        :param samples_per_segment: Data points count for each Welch's method window.
        :param nfft_: Length of the used FFT.
        :return: None.
        """
        self._spectral_power_window = window_width
        self._samples_per_segment = samples_per_segment
        self._nfft = nfft_

    @property
    def power_spectra(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self._power_spectra

    def apply(self, candidates: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spline interpolation/approximation and spectral analysis to a EDR candidates matrix and return processed
        results.
        :param candidates: Matrix of EDR candidates, each candidate occupies an entire row.
        :param timestamps: Markers in seconds of the extracted R peaks. Used to properly reconstruct the timeline of the
        EDR signal.
        :return: Tuple of Interpolated EDR candidates sorted from best to worst and their associated spectral power
        fractions.
        """
        derived_respiration_signals = []
        fractions = []

        extended_domain = np.arange(self._time_boundaries[0], self._time_boundaries[-1], 1 / self._sampling_frequency)

        for edr_row in candidates:
            # Interpolate candidate with cubic spline to restore original sampling frequency and regularity.
            EDR = edr_row.reshape(-1, 1)
            edr_spline = splrep(timestamps, EDR, s=self._spline_smoothing)
            interpolated_edr = splev(extended_domain, edr_spline, der=self._spline_derivative)

            # Transform to frequency domain
            edr_spectrum_domain, edr_spectrum = welch(interpolated_edr, self._sampling_frequency,
                                                      nperseg=self._samples_per_segment,
                                                      nfft=self._nfft)

            self._power_spectra.append((edr_spectrum_domain, edr_spectrum))

            window_width_index = self._spectral_power_window / self._sampling_frequency * 2 * edr_spectrum.shape[0]

            max_index = np.argmax(edr_spectrum)  # Locate global maximum

            # Place window at global maximum with window_width

            left_bracket = int(max_index - window_width_index // 2)
            right_bracket = int(max_index + window_width_index // 2 + 1)

            inside_range = np.r_[left_bracket: right_bracket]
            outside_range = np.r_[0:left_bracket,
                                  right_bracket: edr_spectrum.shape[0]]

            edr_spectrum_inside = edr_spectrum[inside_range]
            edr_spectrum_domain_inside = edr_spectrum_domain[inside_range]

            edr_spectrum_outside = edr_spectrum[outside_range]
            edr_spectrum_domain_outside = edr_spectrum_domain[outside_range]

            # Calculate spectral power inside and outside the window
            spectral_power_inside = simps(edr_spectrum_inside, edr_spectrum_domain_inside)
            spectral_power_outside = simps(edr_spectrum_outside, edr_spectrum_domain_outside)

            # Calculate the outside/inside power fraction
            fractions.append(spectral_power_outside / spectral_power_inside)
            derived_respiration_signals.append(interpolated_edr)

        # Sort EDR candidates with ascending order with respect to the fractions
        zipped_results = zip(fractions, derived_respiration_signals)
        sorted_results = sorted(zipped_results, reverse=False, key=lambda x: x[0])

        sorted_fractions = np.array([el[0] for el in sorted_results])
        sorted_edr = np.array([el[1] for el in sorted_results])

        return sorted_fractions, sorted_edr


if __name__ == "__main__":
    ann = wfdb.io.rdann("../fantasia_wfdb/f1y05", "ecg")
    data = wfdb.io.rdrecord("../fantasia_wfdb/f1y05")

    respiration = data.p_signal[:, 0]
    idxs = ann.sample

    pt_len = 100
    start = 700

    chosen = idxs[start:start + pt_len]
    chosen_time = np.array(chosen) * 1 / 250

    respiration_chosen_raw = np.array(respiration[chosen])

    fig = plt.figure(figsize=(16, 10))

    fig.add_subplot(3, 1, 1)
    plt.plot(chosen_time, respiration_chosen_raw)
    plt.title("Simulated EDR candidate")

    edr_picker = EDRPicker((chosen_time[0], chosen_time[-1]))

    edr_picker.set_spectral_params(.08, 2 ** 13, 2 ** 16)
    edr_picker.set_spline_params(0.1, 0, 250)

    frac, edr = edr_picker.apply(np.array([respiration_chosen_raw]), chosen_time)

    fig.add_subplot(3, 1, 2)
    plt.plot(np.linspace(chosen_time[0], chosen_time[-1], edr[0].shape[0]), edr[0])
    plt.title("Interpolated EDR candidate")

    fig.add_subplot(3, 1, 3)
    respiration_filtered = savgol_filter(respiration[chosen[0]:chosen[-1]], 100, 1)
    plt.plot(np.linspace(chosen_time[0], chosen_time[-1], respiration_filtered.shape[0]), respiration_filtered)
    plt.title("Filtered reference")
    plt.xlabel("Time (s)")

    s = edr_picker.power_spectra

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.plot(s[0][1])

    plt.show()
