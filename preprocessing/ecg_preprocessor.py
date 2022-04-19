import numpy as np
import wfdb
from wfdb import processing


class ECGPreprocessor:
    def __init__(self, signal_path: str):
        self._signal_path = signal_path
        self.record = wfdb.rdrecord(signal_path)
        self.ecg_signal = self.record.p_signal[:, 1]
        self.respiration_signal = self.record.p_signal[:, 0]

    def get_r_peaks_indexes(self) -> np.ndarray:
        """
        Extract QRS complexes from ECG signal

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
        return indexes

    def create_qrs_matrix(self, qrs_time_window: float, data_centering: bool = True) -> np.ndarray:
        """
        Creates matrix with separated QRS complexes and enables optional signal preprocessing

        :param qrs_time_window: Time window in seconds around R peak
        :param data_centering: Enabling data centering by removing the baseline (mean) separately from every QRS segment
        :return: Multivariate matrix of indexes where each column corresponds to R peak
        """
        frequency = self.record.fs
        signal = self.ecg_signal

        indexes = self.get_r_peaks_indexes()
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

        return r_peaks
