from sklearn.decomposition import PCA
from preprocessing import ECGPreprocessor
from postprocessing import EDRPicker
from scipy.signal import savgol_filter, coherence, correlate
import wfdb
import numpy as np
import matplotlib.pyplot as plt

SAMPLING_FREQ = 250
START_IDX = 2251 * SAMPLING_FREQ
SIGNAL_LEN = 5*SAMPLING_FREQ*60

DB_PATH = 'fantasia_wfdb\\f1o01'
ecg_preprocessor = ECGPreprocessor(DB_PATH)
ecg_preprocessor.ecg_signal = savgol_filter(ecg_preprocessor.ecg_signal, window_length=7, polyorder=3)
peaks = ecg_preprocessor.create_qrs_matrix(qrs_time_window=0.12)
idx = ecg_preprocessor.get_r_peaks_indexes()
slicer = np.r_[idx > START_IDX, idx < START_IDX + SIGNAL_LEN]

slicer1 = np.r_[idx > START_IDX]
slicer2 = np.r_[idx < START_IDX + SIGNAL_LEN]

slicer = np.array([a and b for a, b in zip(slicer1, slicer2)])

idx = idx[slicer]
peaks = peaks[:,  slicer]
idx = idx / SAMPLING_FREQ

pca = PCA()
pca.fit(peaks)

eigen_vectors = pca.components_

picker = EDRPicker((idx[0], idx[-1]))
picker.set_spectral_params(window_width=.08,  samples_per_segment=2 ** 13, nfft_=2 ** 16)
picker.set_spline_params(smoothing=0, derivative=0, sampling_frequency=SAMPLING_FREQ)
fractions, edrs = picker.apply(eigen_vectors, idx)
power_spectra = picker.power_spectra

for pair in power_spectra:
    # plt.plot(pair[0], pair[1])
    # plt.show()
    pass


resp = ecg_preprocessor.respiration_signal[START_IDX: START_IDX + SIGNAL_LEN]
# plt.subplot(121)
# plt.plot(resp[:resp.shape[0]//3])
# plt.subplot(122)
# plt.plot(edrs[0][:resp.shape[0]//3])
# plt.show()
print(len(resp), len(edrs[0]))
print(max(correlate(resp, edrs[0])/len(edrs[0])))
print(coherence(resp, edrs[0], fs=ecg_preprocessor.record.fs, nfft=2**16, nperseg=2**13)[0])
# wfdb.plot_items(eigen_vectors[20, :])
# wfdb.plot_items(ecg_preprocessor.respiration_signal)
