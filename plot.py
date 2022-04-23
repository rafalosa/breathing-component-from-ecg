import matplotlib.pyplot as plt
from typing import Tuple, Optional
import numpy as np


def plot_signal(signal: np.ndarray, labels: Tuple[str, str] = ('x', 'y'), title: Optional[str] = None,
                fs: Optional[int] = None):

    if fs is not None:
        t = np.linspace(0, len(signal) - 1, len(signal)) / fs
        plt.plot(t, signal)
    else:
        plt.plot(signal)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.show()
