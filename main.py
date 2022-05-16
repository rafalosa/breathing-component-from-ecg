from scipy.signal import savgol_filter, welch
import numpy as np
import matplotlib.pyplot as plt

from data_processing import ECGPreprocessor, EDRExtractor, EDRPicker
from tools.plot import CollectivePlotter
from tools.signal_testing import coherence, cross_correlation

START_TIME = 2251  # [s]
SIGNAL_LENGTH = 300  # [s]
RANDOM_STATE = 69
PLOTS_NUM = 3

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
picker.set_spline_params(smoothing=0.1, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker.set_spectral_params(window_width=.04, samples_per_segment=2**13, nfft_=2**16)

picker2 = EDRPicker((first_qrs, last_qrs))
picker2.set_spline_params(smoothing=0, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker2.set_spectral_params(window_width=.08, samples_per_segment=2**13, nfft_=2**16)

edr_ica = picker.apply(ica_edr.T, r_timestamps, method="sugpen", crop_to_freq=.6)
ica_spectra = picker.power_spectra

edr_pca = picker2.apply(pca_edr, r_timestamps, method="sugpen", crop_to_freq=.6)
pca_spectra = picker2.power_spectra

pl = CollectivePlotter(3, 4)

edr_domain = np.arange(r_timestamps[0], r_timestamps[-1], 1/SAMPLING_FREQ)

pl.set_col_values(0, edr_ica[:PLOTS_NUM])
pl.set_col_values(2, edr_pca[:PLOTS_NUM])
pl.set_col_values(1, [values[1] for values in ica_spectra[:PLOTS_NUM]])
pl.set_col_values(3, [values[1] for values in pca_spectra[:PLOTS_NUM]])

pl.set_col_domain(0, edr_domain)
pl.set_col_domain(2, edr_domain)
pl.set_col_domain(1, ica_spectra[0][0])
pl.set_col_domain(3, pca_spectra[0][0])

pl.set_col_boundaries(0, (START_TIME, START_TIME + 100))
pl.set_col_boundaries(2, (START_TIME, START_TIME + 100))

pl.set_col_xlabel(0, "Time (s)")
pl.set_col_xlabel(1, "Frequency (Hz)")
pl.set_col_xlabel(2, "Time (s)")
pl.set_col_xlabel(3, "Frequency (Hz)")
pl.show()

# Test section
for i in range(3):
    edr_pca_to_test = edr_pca[i]
    edr_ica_to_test = edr_ica[i]
    edr_length = len(edr_pca_to_test)
    resp_signal = savgol_filter(ecg_preprocessor.respiration_signal, 150, 2)[:edr_length]

    pca_cross_corr = cross_correlation(resp_signal, edr_pca_to_test, adjusted=False)
    ica_cross_corr = cross_correlation(resp_signal, edr_ica_to_test, adjusted=False)

    pca_coherence = coherence(resp_signal, edr_pca_to_test, fs=SAMPLING_FREQ, window_type='hamming',
                              samples_per_segment=2 ** 13, nfft=2 ** 16)
    ica_coherence = coherence(resp_signal, edr_ica_to_test, fs=SAMPLING_FREQ, window_type='hamming',
                              samples_per_segment=2 ** 13, nfft=2 ** 16)

    print(f'{i + 1} - PCA cross correlation and coherence results: {pca_cross_corr:.2f}, {pca_coherence:.2f}')
    print(f'{i + 1} - ICA cross correlation and coherence results: {ica_cross_corr:.2f}, {ica_coherence:.2f}')

pl2 = CollectivePlotter(1, 2)

pl2.set_col_values(0, [savgol_filter(ecg_preprocessor.respiration_signal, 150, 2)])
pl2.set_col_domain(0, np.linspace(START_TIME, START_TIME+SIGNAL_LENGTH, SIGNAL_LENGTH*SAMPLING_FREQ))
pl2.set_col_boundaries(0, (START_TIME, START_TIME + 100))

domain, respiration_spectrum = welch(ecg_preprocessor.respiration_signal, fs=SAMPLING_FREQ, nperseg=2 ** 13, nfft=2 ** 16)
pl2.set_col_values(1, [respiration_spectrum])
pl2.set_col_domain(1, domain)
pl2.set_col_boundaries(1, (0, .6))

pl2.show()

