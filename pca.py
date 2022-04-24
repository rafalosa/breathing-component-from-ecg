from sklearn.decomposition import PCA
from data_processing import ECGPreprocessor, EDRPicker
from scipy.signal import savgol_filter, welch
from tools.plot import CollectivePlotter
import wfdb
import numpy as np
import matplotlib.pyplot as plt

START_TIME = 2651  # [s]
SIGNAL_LENGTH = 300  # [s]
SAMPLING_FREQ = 250  # [Hz]

start_idx = START_TIME * SAMPLING_FREQ
end_idx = (START_TIME + SIGNAL_LENGTH) * SAMPLING_FREQ

DB_PATH = 'data\\fantasia_wfdb\\f1o01'

ecg_preprocessor = ECGPreprocessor(DB_PATH)
ecg_preprocessor.ecg_signal = savgol_filter(ecg_preprocessor.ecg_signal, window_length=7, polyorder=3)

r_idxs = ecg_preprocessor.get_r_peaks_indexes()
ecg_signal = ecg_preprocessor.ecg_signal
qrs_complexes_all = ecg_preprocessor.create_qrs_matrix(qrs_time_window=0.12)
resp_signal = ecg_preprocessor.respiration_signal
resp_signal_cropped = resp_signal[start_idx:end_idx]


slicer_temp1 = np.r_[r_idxs > start_idx]
slicer_temp2 = np.r_[r_idxs < end_idx]
slicer = [condA and condB for condA, condB in zip(slicer_temp1, slicer_temp2)]

r_timestamps_cropped = r_idxs[slicer] / 250
qrs_complexes_cropped = qrs_complexes_all[:, slicer]

pca = PCA()
pca.fit(qrs_complexes_cropped)

eigen_vectors = pca.components_

picker = EDRPicker((r_timestamps_cropped[0], r_timestamps_cropped[-1]))
picker.set_spline_params(smoothing=0, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker.set_spectral_params(window_width=.08, samples_per_segment=2**13, nfft_=2**16)

fractions, edrs = picker.apply(eigen_vectors, r_timestamps_cropped, sort_=True)
edr_domain = np.arange(r_timestamps_cropped[0], r_timestamps_cropped[-1], 1/SAMPLING_FREQ)

EDR_IDX = 0

domain = picker.power_spectra[EDR_IDX][0]

pl = CollectivePlotter(3, 2)
edrs = list(edrs)
spectra = [tup[1] for tup in picker.power_spectra]

pl.set_col_values(0, edrs[:3])
pl.set_col_values(1, spectra[:3])
pl.set_col_domain(0, edr_domain)
pl.set_col_domain(1, domain)
pl.set_col_boundaries(1, (0, .5))
pl.set_col_boundaries(0, (START_TIME, START_TIME + 100))

pl.set_col_xlabel(0, "Time (s)")
pl.set_col_xlabel(1, "Frequency (Hz)")
# titles = ['PCA1', 'PCA2', 'PCA3']
pl.show((10, 5))#, titles=titles)
