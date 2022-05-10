from data_processing.ecg_preprocessor import ECGPreprocessor
from data_processing.edr_picker import EDRPicker
import numpy as np
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tools import CollectivePlotter

START_TIME = 2651  # [s]
SIGNAL_LENGTH = 300  # [s]
SAMPLING_FREQ = 250  # [Hz]

start_idx = START_TIME * SAMPLING_FREQ
end_idx = (START_TIME + SIGNAL_LENGTH) * SAMPLING_FREQ

extractor = ECGPreprocessor("data/fantasia_wfdb/f1o01", START_TIME, START_TIME + SIGNAL_LENGTH)

r_timestamps = extractor.get_r_peaks_indexes()/SAMPLING_FREQ
ecg_signal = extractor.ecg_signal
qrs_complexes = extractor.create_qrs_matrix(.12)
resp_signal = extractor.respiration_signal

pca = PCA(n_components=6, whiten=True, random_state=123)
pca.fit(qrs_complexes)
reduced_dim_data = pca.components_

ica = FastICA(max_iter=160, tol=.1, random_state=123)
components = ica.fit_transform(reduced_dim_data.T)

picker = EDRPicker((r_timestamps[0], r_timestamps[-1]))
picker.set_spline_params(smoothing=0, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker.set_spectral_params(window_width=.08, samples_per_segment=2**13, nfft_=2**16)

fractions, edrs = picker.apply(components.T, r_timestamps, crop_to_freq=.6)
edr_domain = np.arange(r_timestamps[0], r_timestamps[-1], 1/SAMPLING_FREQ)

fig = plt.figure(figsize=(10, 5))

EDR_IDX = 0

fig.add_subplot(3, 1, 1)
plt.plot(edrs[EDR_IDX][:(SIGNAL_LENGTH*250)//3]*-1)

fig.add_subplot(3, 1, 2)
plt.plot(savgol_filter(resp_signal[:(SIGNAL_LENGTH*250)//3], 100, 1))

fig.add_subplot(3, 1, 3)
domain = picker.power_spectra[EDR_IDX][0]
spectrum = picker.power_spectra[EDR_IDX][1]
plt.plot(domain, spectrum)

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






