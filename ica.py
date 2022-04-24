from preprocessing.ecg_preprocessor import ECGPreprocessor
from postprocessing.edr_picker import EDRPicker
import numpy as np
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from plot import CollectivePlotter

START_TIME = 2651  # [s]
SIGNAL_LENGTH = 300  # [s]
SAMPLING_FREQ = 250  # [Hz]

start_idx = START_TIME * SAMPLING_FREQ
end_idx = (START_TIME + SIGNAL_LENGTH) * SAMPLING_FREQ

extractor = ECGPreprocessor("data/fantasia_wfdb/f1o01")

r_idxs = extractor.get_r_peaks_indexes()
ecg_signal = extractor.ecg_signal
qrs_complexes_all = extractor.create_qrs_matrix(.12)
resp_signal = extractor.respiration_signal
resp_signal_cropped = resp_signal[start_idx:end_idx]

slicer_temp1 = np.r_[r_idxs > start_idx]
slicer_temp2 = np.r_[r_idxs < end_idx]
slicer = [condA and condB for condA, condB in zip(slicer_temp1, slicer_temp2)]

r_timestamps_cropped = r_idxs[slicer] / 250
qrs_complexes_cropped = qrs_complexes_all[:, slicer]

pca = PCA(n_components=6, whiten=True, random_state=123)
pca.fit(qrs_complexes_cropped)
reduced_dim_data = pca.components_

ica = FastICA(max_iter=160, tol=.1, random_state=123)
components = ica.fit_transform(reduced_dim_data.T)

picker = EDRPicker((r_timestamps_cropped[0], r_timestamps_cropped[-1]))
picker.set_spline_params(smoothing=0, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker.set_spectral_params(window_width=.08, samples_per_segment=2**13, nfft_=2**16)

fractions, edrs = picker.apply(components.T, r_timestamps_cropped)
edr_domain = np.arange(r_timestamps_cropped[0], r_timestamps_cropped[-1], 1/SAMPLING_FREQ)

fig = plt.figure(figsize=(10, 5))

EDR_IDX = 0

fig.add_subplot(3, 1, 1)
plt.plot(edrs[EDR_IDX][:(SIGNAL_LENGTH*250)//3]*-1)

fig.add_subplot(3, 1, 2)
plt.plot(savgol_filter(resp_signal_cropped[:(SIGNAL_LENGTH*250)//3], 100, 1))

fig.add_subplot(3, 1, 3)
domain = picker.power_spectra[EDR_IDX][0]
cropper1 = np.r_[domain >= 0]
cropper2 = np.r_[domain <= .6]
cropper = [a and b for a, b in zip(cropper1, cropper2)]
cropped_domain = domain[cropper]
cropped_spectrum = picker.power_spectra[EDR_IDX][1][cropper]
plt.plot(cropped_domain, cropped_spectrum)

plt.show()

pl = CollectivePlotter(6, 2)
edrs = [edr for edr in edrs]
spectra = [tup[1] for tup in picker.power_spectra]

pl.set_col_values(0, edrs)
pl.set_col_values(1, spectra)
pl.set_col_domain(0, edr_domain)
pl.set_col_domain(1, domain)
pl.set_col_boundaries(1, (0, .5))
pl.set_col_boundaries(0, (START_TIME, START_TIME + 100))

pl.set_col_xlabel(0, "Time (s)")
pl.set_col_xlabel(1, "Frequency (Hz)")

pl.show()






