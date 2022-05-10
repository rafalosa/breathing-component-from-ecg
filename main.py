import wfdb
from scipy.signal import savgol_filter

from data_processing import ECGPreprocessor
from tools.plot import plot_signal

DB_PATH = 'data/fantasia_wfdb/f1o01'
ecg_preprocessor = ECGPreprocessor(DB_PATH)
fs = ecg_preprocessor.record.fs
signal_range = 2000


peaks = ecg_preprocessor.create_qrs_matrix(qrs_time_window=0.12, data_centering=False)
plot_signal(peaks[:, 0], fs=fs, labels=('Time(s)', 'Signal(a.u)'), title='Raw signal')
# plot_signal(ecg_preprocessor.ecg_signal[:signal_range], fs=fs, labels=('Time(s)', 'Signal(a.u)'))
# plot_signal(ecg_preprocessor.ecg_signal, fs=fs, labels=('Time(s)', 'Signal(a.u)'))
plot_signal(peaks.T.flatten()[:signal_range//5], fs=fs, labels=('Samples', 'Signal(a.u)'))


# ecg_preprocessor.ecg_signal = savgol_filter(ecg_preprocessor.ecg_signal, window_length=7, polyorder=3)
peaks = ecg_preprocessor.create_qrs_matrix(qrs_time_window=0.12, data_centering=True)
# plot_signal(peaks[:, 0], fs=fs, labels=('Time(s)', 'Signal(a.u)'))
# plot_signal(ecg_preprocessor.ecg_signal[:signal_range], fs=fs)
# plot_signal(ecg_preprocessor.ecg_signal, fs=fs)
plot_signal(peaks.T.flatten()[:signal_range//5], fs=fs, labels=('Samples', 'Signal(a.u)'))
