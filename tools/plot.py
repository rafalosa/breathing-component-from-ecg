import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
import numpy as np
from functools import wraps


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


class CollectivePlotter:
    __slots__ = {"_values",
                 "_col_domains",
                 "_col_boundaries",
                 "_ncols",
                 "_nrows",
                 "_xlabels",
                 "_ylabels"
                 }

    def __init__(self, nrows: int, ncols: int):

        self._ncols: int = ncols
        self._nrows: int = nrows
        self._values: List[Optional[np.ndarray]] = [None for _ in range(nrows*ncols)]
        self._col_domains: List[Optional[np.ndarray]] = [None for _ in range(ncols)]
        self._col_boundaries: List[Optional[Tuple[float, float]]] = [None for _ in range(ncols)]
        self._xlabels: List[Optional[str]] = [None for _ in range(ncols)]
        self._ylabels: List[Optional[str]] = [None for _ in range(ncols)]

    def col_boundaries(self, ncol: int) -> Tuple[float, float]:

        """
        Get X axis boundaries for the specified columns of the plot

        :param ncol: Specified column index
        :return: None
        """

        if ncol >= self._ncols:
            raise IndexError(f"Column of index {ncol} is out of bounds for number of columns: {self._ncols}")

        return self._col_boundaries[ncol]

    def set_col_boundaries(self, ncol: int, boundaries: Tuple[float, float]) -> None:
        """
        Set X axis boundaries for the specified columns of the plot. The provided domain data will be cropped to fit
        the boundaries

        :param ncol: Column index to which the boundaries will be applied
        :param boundaries: Left and right boundary
        :return: None
        """

        if boundaries[0] >= boundaries[1]:
            raise RuntimeError("Right side boundary has to be bigger.")

        if ncol >= self._ncols:
            raise IndexError(f"Column of index {ncol} is out of bounds for number of columns: {self._ncols}")

        self._col_boundaries[ncol] = boundaries

    def set_col_xlabel(self, ncol: int, label: str) -> None:
        """
        Set the X axis label for the specified column.

        :param ncol: Column index to which the x label will be applied
        :param label: Label string
        :return: None
        """

        if ncol >= self._ncols:
            raise IndexError(f"Column of index {ncol} is out of bounds for number of columns: {self._ncols}")

        self._xlabels[ncol] = label

    def set_col_ylabel(self, ncol: int, label: str) -> None:
        """
        Set the Y axis label for the specified column.

        :param ncol: Column index to which the Y label will be applied
        :param label: Label string
        :return: None
        """
        if ncol >= self._ncols:
            raise IndexError(f"Column of index {ncol} is out of bounds for number of columns: {self._ncols}")

        self._ylabels[ncol] = label

    def set_col_values(self, ncol: int, values: Union[List[np.ndarray], np.ndarray]) -> None:
        """
        Set the Y axis data for all plots in the specified column

        :param ncol: Specified column index
        :param values: Values to be displayed as Y data.
        :return: None
        """

        if ncol >= self._ncols:
            raise IndexError(f"Column of index {ncol} is out of bounds for number of columns: {self._ncols}")

        if len(values) != self._nrows:
            raise RuntimeError(f"Each column has {self._nrows} records, {len(values)} were given.")

        for j, col_values in enumerate(values):
            self._values[ncol + j*self._ncols] = col_values

    def set_col_domain(self, ncol: int, domain: np.ndarray) -> None:
        """
        Set the X axis data for all plots in the specified column. All plots in one column share the same domain

        :param ncol: Specified column index
        :param domain: Array of values to be used as column domain
        :return: None
        """
        if ncol >= self._ncols:
            raise IndexError(f"Column of index {ncol} is out of bounds for number of columns: {self._ncols}")

        self._col_domains[ncol] = domain

    def show(self, figsize_: Tuple[int, int] = (10, 10)) -> None:
        """
        Display the plot with current settings

        :param figsize_: Size of the displayed figure
        :return: None
        """

        fig = plt.figure(figsize=figsize_)

        for i in range(self._ncols):
            overlay_ax = fig.add_subplot(1, self._ncols, i+1)
            overlay_ax.spines['top'].set_color('none')
            overlay_ax.spines['bottom'].set_color('none')
            overlay_ax.spines['left'].set_color('none')
            overlay_ax.spines['right'].set_color('none')
            overlay_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            
            xlabel = self._xlabels[i]
            ylabel = self._ylabels[i]
            
            if xlabel is not None:
                overlay_ax.set_xlabel(xlabel, fontsize=8, labelpad=7)
            if ylabel is not None:
                overlay_ax.set_ylabel(ylabel, fontsize=8, labelpad=7)

        for i in range(self._ncols * self._nrows):

            col_idx = i % self._ncols
            domain = self._col_domains[col_idx]
            values = self._values[i]

            if self._col_boundaries[col_idx] is not None:
                left_b, right_b = self._col_boundaries[col_idx]
                slicer_temp1 = np.r_[domain > left_b]
                slicer_temp2 = np.r_[domain < right_b]
                slicer = [a and b for a, b in zip(slicer_temp1, slicer_temp2)]
                domain = domain[slicer]
                values = values[slicer]

            ax = fig.add_subplot(self._nrows, self._ncols, i+1)
            if domain is not None:
                ax.plot(domain, values)
            else:
                ax.plot(values)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
        plt.show()


if __name__ == "__main__":
    pl = CollectivePlotter(3, 3)
    dom = np.linspace(0, np.pi, 100)

    for freq_idx in range(3):
        k = 5 * (freq_idx + 1)

        vals = [np.sin(2 * dom * k), np.sin(4 * dom * k), np.sin(6 * dom * k)]
        pl.set_col_domain(freq_idx, dom)
        pl.set_col_values(freq_idx, vals)

    pl.show(figsize_=(15, 15))
