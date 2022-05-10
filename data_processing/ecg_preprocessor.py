from typing import Optional

import numpy as np
import wfdb
from wfdb import processing


class ECGPreprocessor:
    def __init__(self, signal_path: str, start: Optional[float] = None, end: Optional[float] = None):
        self._signal_path = signal_path
        self.record = wfdb.rdrecord(signal_path)
        self._ecg_signal = self.record.p_signal[:, 1]
        self._respiration_signal = self.record.p_signal[:, 0]
        self.start_time: Optional[float] = start
        self.end_time: Optional[float] = end
        self.signal_length: Optional[float] = None

        if self.start_time is not None and self.end_time is not None:
            self.signal_length = self.end_time - self.start_time

    @property
    def ecg_signal(self):

        if self.start_time is not None and self.end_time is not None:

            start_idx = self.start_time * self.record.fs
            end_idx = self.end_time * self.record.fs
            return self._ecg_signal[start_idx:end_idx]

        return self._ecg_signal

    @property
    def respiration_signal(self):

        if self.start_time is not None and self.end_time is not None:

            start_idx = self.start_time * self.record.fs
            end_idx = self.end_time * self.record.fs
            return self._respiration_signal[start_idx:end_idx]

        return self._respiration_signal

    def get_r_peaks_indexes(self, cropped: bool = True) -> np.ndarray:
        """
        Extract QRS complexes from ECG signal

        :param cropped: Return R peaks occurring in the specified time period.
        :return: Indices of the detected QRS complexes
        """
        try:
            indexes = wfdb.rdann(self._signal_path, 'ecg').sample

        except FileNotFoundError:
            try:
                indexes = wfdb.rdann(self._signal_path, 'atr').sample
            except FileNotFoundError:
                xqrs = processing.XQRS(sig=self.ecg_signal, fs=self.record.fs)
                xqrs.detect()
                indexes = xqrs.qrs_inds

        if self.start_time is not None and self.end_time is not None and cropped:
            slicer = self._index_slicer(indexes=indexes)
            indexes = indexes[slicer]

        return indexes

    def create_qrs_matrix(self, qrs_time_window: float, data_centering: bool = True) -> np.ndarray:
        """
        Creates matrix with separated QRS complexes and enables optional signal preprocessing

        :param qrs_time_window: Time window in seconds around R peak
        :param data_centering: Enabling data centering by removing the baseline (mean) separately from every QRS segment
        :return: Multivariate matrix of indexes where each column corresponds to R peak
        """
        frequency = self.record.fs
        signal = self._ecg_signal

        indexes = self.get_r_peaks_indexes(cropped=False)
        max_window_width = min([y - x for x, y in zip(indexes, indexes[1:])])

        number_of_samples = int(frequency * qrs_time_window)

        if number_of_samples > max_window_width:
            raise ValueError('Value exceed max window time')

        r_peaks = np.zeros((number_of_samples, len(indexes)))

        for n, index in enumerate(indexes):
            start = int(index - number_of_samples / 2)
            end = int(index + number_of_samples / 2)
            r_peaks[:, n] = signal[start:end]

        if data_centering:
            r_peaks = r_peaks - r_peaks.mean(axis=0)

        if self.start_time is not None and self.end_time is not None:
            slicer = self._index_slicer(indexes=indexes)
            r_peaks = r_peaks[:, slicer]

        return r_peaks

    def _index_slicer(self, indexes: np.ndarray) -> np.r_:

        # todo: Improve

        start_idx = self.start_time * self.record.fs
        end_idx = self.end_time * self.record.fs
        slicer_temp1 = np.r_[indexes > start_idx]
        slicer_temp2 = np.r_[indexes < end_idx]
        slicer = [condA and condB for condA, condB in zip(slicer_temp1, slicer_temp2)]

        return slicer
