from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import wfdb.io
from scipy.integrate import simps
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter
from scipy.signal import welch


class EDRPicker:

    __slots__ = {"_spectral_power_window",
                 "_sampling_frequency",
                 "_spline_smoothing",
                 "_spline_derivative",
                 "_time_boundaries",
                 "_samples_per_segment",
                 "_nfft",
                 "candidates_raw",
                 "_power_spectra",
                 "_power_spectra_cropped",
                 "_cropped",
                 "_freq_suggest"}

    def __init__(self, time_boundaries: Tuple[float, float]):
        self._time_boundaries = time_boundaries
        self._spline_smoothing = 0
        self._spline_derivative = 0
        self._sampling_frequency = 100
        self._spectral_power_window = .1
        self._samples_per_segment = 2 ** 10
        self._nfft = 2 ** 10
        self._power_spectra: List[Tuple[np.ndarray, np.ndarray]] = []
        self._power_spectra_cropped: List[Tuple[np.ndarray, np.ndarray]] = []
        self._freq_suggest: Optional[float] = None
        self._cropped: bool = False

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

    def set_spectral_params(self, window_width: float = .1,
                            samples_per_segment: int = 2 ** 10,
                            nfft_: int = 2 ** 10,
                            suggest_respiration_freq: float = .25) -> None:
        """
        Set spectral analysis parameters.
        :param window_width: Window width in Hz around the global maximum/maxima of the spectrum for each EDR candidate.
        :param samples_per_segment: Data points count for each Welch's method window.
        :param nfft_: Length of the used FFT.
        :param suggest_respiration_freq: Frequency peak around which the respiration peak will me searched for.
        :return: None.
        """
        self._spectral_power_window = window_width
        self._samples_per_segment = samples_per_segment
        self._nfft = nfft_
        self._freq_suggest = suggest_respiration_freq

    @property
    def power_spectra(self) -> List[Tuple[np.ndarray, np.ndarray]]:

        if not self._power_spectra:
            raise RuntimeError("Perform the spectral analysis using the apply()"
                               " method before getting the candidates' spectra.")
        if self._cropped:
            return self._power_spectra_cropped
        else:
            return self._power_spectra

    def apply(self, candidates: np.ndarray,
              timestamps: np.ndarray,
              method: Optional[str] = "pwfr",
              crop_to_freq: Optional[float] = None) -> np.ndarray:
        """
        Apply spline interpolation/approximation and spectral analysis to a EDR candidates matrix and return processed
        results.
        :param candidates: Matrix of EDR candidates, each candidate occupies an entire row.
        :param timestamps: Markers in seconds of the extracted R peaks. Used to properly reconstruct the timeline of the
        EDR signal.
        :param method: Sorting method of the extracted EDR candidates.
        :param crop_to_freq: Frequency to which the power spectra and their domains will be cropped.
        :return: Tuple of Interpolated EDR candidates sorted from best to worst and their associated spectral power
        fractions.
        """
        if crop_to_freq is not None:
            self._cropped = True

        self._power_spectra = []
        interpolated_edrs = []

        if method is not None:
            method = method.lower()

        extended_domain = np.arange(self._time_boundaries[0], self._time_boundaries[-1], 1 / self._sampling_frequency)

        for edr_row in candidates:
            # Interpolate candidate with cubic spline to restore original sampling frequency and regularity.
            EDR = edr_row.reshape(-1, 1)
            edr_spline = splrep(timestamps, EDR, s=self._spline_smoothing)
            interp_edr = splev(extended_domain, edr_spline, der=self._spline_derivative)
            interpolated_edrs.append(interp_edr)

            # Transform to frequency domain
            edr_spectrum_domain, edr_spectrum = welch(interp_edr, self._sampling_frequency,
                                                      nperseg=self._samples_per_segment,
                                                      nfft=self._nfft)
            self._power_spectra.append((edr_spectrum_domain, edr_spectrum))

            if self._cropped:
                # todo: Improve
                cropper1 = np.r_[edr_spectrum_domain >= 0]
                cropper2 = np.r_[edr_spectrum_domain <= crop_to_freq]
                cropper = [cond1 and cond2 for cond1, cond2 in zip(cropper1, cropper2)]
                cropped_edr_domain = edr_spectrum_domain[cropper]
                cropped_edr_spectrum = edr_spectrum[cropper]
                self._power_spectra_cropped.append((cropped_edr_domain, cropped_edr_spectrum))

        if method == "pwfr":
            derived_respiration_signals = np.array(self._power_fraction_picking(interpolated_edrs))
        elif method == "sugpen":
            derived_respiration_signals = np.array(self._approx_freq_picking(interpolated_edrs))
        else:
            derived_respiration_signals = interpolated_edrs

        return derived_respiration_signals

    def _power_fraction_picking(self, interpolated_edrs: List[np.ndarray]):

        fractions = []
        derived_respiration_signals = interpolated_edrs

        for edr_spectrum_domain, edr_spectrum in self._power_spectra:

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

        # Sort EDR candidates with ascending order with respect to the fractions
        derived_respiration_signals = self._sort_candidates_and_spectra(fractions,
                                                                        derived_respiration_signals,
                                                                        False)

        return derived_respiration_signals

    def _approx_freq_picking(self, interpolated_edrs: List[np.ndarray]):

        derived_respiration_signals = interpolated_edrs
        penalties = []
        spectral_penalties = []
        distance_penalties = []

        for edr_spectrum_domain, edr_spectrum in self._power_spectra:

            window_width_index = self._spectral_power_window / self._sampling_frequency * 2 * edr_spectrum.shape[0]

            max_index = np.argmax(edr_spectrum)
            max_freq = edr_spectrum_domain[max_index]

            left_bracket = int(max_index - window_width_index // 2)
            right_bracket = int(max_index + window_width_index // 2 + 1)

            inside_range = np.r_[left_bracket: right_bracket]

            edr_spectrum_inside = edr_spectrum[inside_range]
            edr_spectrum_domain_inside = edr_spectrum_domain[inside_range]

            spectral_penalties.append(simps(edr_spectrum_inside, edr_spectrum_domain_inside))
            distance_penalties.append(np.abs(self._freq_suggest - max_freq))

        distance_penalties = np.array(distance_penalties)
        spectral_penalties = np.array(spectral_penalties)

        distance_penalty_max = max(distance_penalties)
        spectral_penalty_max = max(spectral_penalties)

        distance_penalties /= distance_penalty_max
        spectral_penalties /= spectral_penalty_max

        for dist_pen, spec_pen in zip(distance_penalties, spectral_penalties):

            penalties.append(dist_pen + spec_pen)

        derived_respiration_signals = self._sort_candidates_and_spectra(penalties,
                                                                        derived_respiration_signals,
                                                                        False)

        return derived_respiration_signals

    def _sort_candidates_and_spectra(self, sort_args: List[float], edrs: List[np.ndarray], reverse: bool):

        zipped_results = zip(sort_args, edrs, self._power_spectra)

        if self._cropped:
            zipped_results = zip(sort_args, edrs, self._power_spectra, self._power_spectra_cropped)

        zipped_results = sorted(zipped_results, reverse=reverse, key=lambda x: x[0])
        arg = np.array([el[1] for el in zipped_results])
        derived_respiration_signals = np.array([el[1] for el in zipped_results])
        self._power_spectra = [el[2] for el in zipped_results]
        if self._cropped:
            self._power_spectra_cropped = [el[3] for el in zipped_results]

        return derived_respiration_signals


if __name__ == "__main__":
    pass
