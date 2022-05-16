from typing import Optional

import numpy as np
from scipy import signal
from statsmodels.tsa.stattools import ccf


def cross_correlation(x: np.ndarray, y: np.ndarray, adjusted: bool = True) -> float:
    """
    Calculate normalized crossed correlation between to signals x and y

    :param x: Signal X
    :param y: Signal Y
    :param adjusted: If True, then denominators for cross-correlation is n-k, otherwise n.
    :return: Maximum value of cross correlation representing correlation coefficient of the data
    """
    return max(ccf(x, y, adjusted=adjusted))


def coherence(x: np.ndarray, y: np.ndarray, fs: float, window_type: Optional[str] = 'hamming',
              samples_per_segment: Optional[int] = None, nfft: Optional[int] = None) -> float:
    """
    Calcalate the maximum of magnitude squared coherence modulus

    :param x: Signal X
    :param y: Signal Y
    :param fs: Sampling frequency of the x and y time series
    :param window_type: The type of window to create
    :param samples_per_segment: The number of samples in the window
    :param nfft: Length of the FFT used
    :return: Coherence
    """
    if samples_per_segment is not None:
        _, pca_coherence = signal.coherence(x, y, fs=fs, window=window_type, nperseg=samples_per_segment, nfft=nfft)
    else:
        _, pca_coherence = signal.coherence(x, y, fs=fs, window=window_type, nperseg=samples_per_segment, nfft=nfft)

    return max(pca_coherence)
