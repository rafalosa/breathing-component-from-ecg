import wfdb
from scipy.signal import savgol_filter

from preprocessing import ECGPreprocessor

DB_PATH = 'fantasia_wfdb\\f1o01'
ecg_preprocessor = ECGPreprocessor(DB_PATH)

wfdb.plot_items(ecg_preprocessor.ecg_signal[:400])
ecg_preprocessor.ecg_signal = savgol_filter(ecg_preprocessor.ecg_signal, window_length=7, polyorder=3)
wfdb.plot_items(ecg_preprocessor.ecg_signal[:400])

peaks = ecg_preprocessor.create_qrs_matrix(qrs_time_window=0.12, data_centering=False)
wfdb.plot_items(peaks[:, 0])
wfdb.plot_items(peaks.T.flatten()[:400])

peaks = ecg_preprocessor.create_qrs_matrix(qrs_time_window=0.12)
wfdb.plot_items(peaks[:, 0])
wfdb.plot_items(peaks.T.flatten()[:400])