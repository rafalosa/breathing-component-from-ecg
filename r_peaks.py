from typing import Callable, Optional

import numpy as np
import wfdb
from scipy.signal import savgol_filter
from wfdb import processing


# TODO: Optionally in the future change it to class
def get_qrs_indexes(record: wfdb.Record) -> np.ndarray:
    """
    Extract QRS complexes from ECG signal

    :param record: ECG record
    :return: Indices of the detected QRS complexes
    """
    ecg = record.p_signal[:, 1]
    xqrs = processing.XQRS(sig=ecg, fs=record.fs)
    xqrs.detect()
    return xqrs.qrs_inds


def ecg_to_qrs_matrix(ecg_file: str, qrs_time: float, data_centering: bool = True,
                      filter_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """
    Creates matrix with separated QRS complexes and enables optional signal preprocessing

    :param ecg_file: File path
    :param qrs_time: Time window in seconds around R peak
    :param data_centering: Enabling data centering by removing the baseline (mean) separately from every QRS segment
    :param filter_fun: Filtering function for noise removal
    :return: Multivariate matrix of indexes where each column corresponds to R peak
    """
    record = wfdb.rdrecord(ecg_file)
    frequency = record.fs
    signal = record.p_signal[:, 1]

    if filter_fun is not None:
        signal = filter_fun(signal)

    if data_centering:
        signal = signal - signal.mean(axis=0)

    try:
        indexes = wfdb.rdann(ecg_file, 'ecg').sample
    except FileNotFoundError:
        indexes = get_qrs_indexes(record)

    number_of_samples = int(frequency * qrs_time)
    r_peaks = np.zeros((number_of_samples, len(indexes)))

    for n, index in enumerate(indexes):
        start = int(index - number_of_samples / 2)
        end = int(index + number_of_samples / 2)
        r_peaks[:, n] = signal[start:end]

    return r_peaks


if __name__ == "__main__":
    DB_PATH = 'fantasia_wfdb/f1o01'

    r = ecg_to_qrs_matrix(DB_PATH, 0.12,
                          filter_fun=lambda signal: savgol_filter(signal, window_length=7, polyorder=3))
    wfdb.plot_items(r[:, 1])
