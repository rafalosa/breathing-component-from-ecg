from scipy.signal import savgol_filter
import numpy as np

from data_processing import ECGPreprocessor, EDRExtractor, EDRPicker
from tools.plot import CollectivePlotter
from tools.signal_testing import coherence, cross_correlation

START_TIME = 2251  # [s]
SIGNAL_LENGTH = 300  # [s]
RANDOM_STATE = 69

DB_PATH = 'data/fantasia_wfdb/f1o01'
ecg_preprocessor = ECGPreprocessor(DB_PATH,
                                   START_TIME,
                                   START_TIME+SIGNAL_LENGTH,
                                   lambda x: savgol_filter(x, window_length=7, polyorder=3))

SAMPLING_FREQ = ecg_preprocessor.record.fs  # [Hz]
r_timestamps = ecg_preprocessor.get_r_peaks_indexes()/SAMPLING_FREQ

first_qrs = r_timestamps[0]
last_qrs = r_timestamps[-1]

qrs_matrix = ecg_preprocessor.create_qrs_matrix(qrs_time_window=.12, data_centering=True)

extractor = EDRExtractor(qrs_matrix)

pca_edr = extractor.pca(random_state=RANDOM_STATE)

ica_edr = extractor.ica(reduce_dim=6, max_iter=160, tol=.1, random_state=RANDOM_STATE)

picker = EDRPicker((first_qrs, last_qrs))
picker.set_spline_params(smoothing=0, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker.set_spectral_params(window_width=.08, samples_per_segment=2**13, nfft_=2**16)

edr_ica = picker.apply(ica_edr.T, r_timestamps, method="sugpen", crop_to_freq=.6)
edr_pca = picker.apply(pca_edr, r_timestamps, method="sugpen", crop_to_freq=.6)

pl = CollectivePlotter(6, 2)

edr_domain = np.arange(r_timestamps[0], r_timestamps[-1], 1/SAMPLING_FREQ)

pl.set_col_values(0, edr_ica[:6])
pl.set_col_values(1, edr_pca[:6])
pl.set_col_domain(0, edr_domain)
pl.set_col_domain(1, edr_domain)
pl.set_col_boundaries(1, (START_TIME, START_TIME + 100))
pl.set_col_boundaries(0, (START_TIME, START_TIME + 100))

pl.set_col_xlabel(0, "Time (s)")
pl.set_col_xlabel(1, "Time (s)")

pl.show()

# Test section
edr_pca_to_test = edr_pca[0]
edr_ica_to_test = edr_ica[0]
edr_length = len(edr_pca_to_test)
resp_signal = ecg_preprocessor.respiration_signal[:edr_length]

pca_cross_corr = cross_correlation(resp_signal, edr_pca_to_test, adjusted=True)
ica_cross_corr = cross_correlation(resp_signal, edr_ica_to_test, adjusted=True)

pca_coherence = coherence(resp_signal, edr_pca_to_test, fs=SAMPLING_FREQ, window_type='hamming', n=2**13, nfft=2**16)
ica_coherence = coherence(resp_signal, edr_ica_to_test, fs=SAMPLING_FREQ, window_type='hamming', n=2**13, nfft=2**16)

print(f'PCA cross correlation and coherence results: {pca_cross_corr:.2f}, {pca_coherence:.2f}')
print(f'ICA cross correlation and coherence results: {ica_cross_corr:.2f}, {ica_coherence:.2f}')
