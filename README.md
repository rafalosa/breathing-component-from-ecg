[rafalosa_repo]: https://gitlab.com/rafalosa
[mkoruszowic_repo]: https://gitlab.com/mkoruszowic

# Breathing component separation
This repository contains a simple set of tools designed for the purpose of breathing component separation from single
channel ECG signal. The set of tools consists of a `data_processing` package, which contains three separate modules for the
three distinct stages in the data processing stage:
* Preprocessing (data upload, filtration, feature extraction)
* EDR extraction (PCA, ICA)
* Postprocessing (EDR interpolation, candidate selection by spectral analysis)

A separate `tools` package is provided with a set of additional tools for creating group plots for easier signal comparisons
or tools for EDR validation.

# Installing dependencies
To ensure correct behaviour of the provided toolset, we recommend installing all the dependencies in a `python 3.8` virtual
environment, using the `requirements.txt` file. To do so, execute the following commands in the project root directory.

### Linux
```bash
python -m venv venv_name
source venv_name/bin/activate
pip install -r requirements.txt
```
### Windows
```
python -m venv venv_name
.\venv_name\Scripts\activate
pip install -r requirements.txt
```
Make sure that the `python` alias leads to the correct version of python executable, specified in the `PATH` environmental variable.

# Workflow
### Preprocessing
The ECG signal processing can be divided into three independent stages, as mentioned above. The first step after downloading the
desired dataset is to upload it to the `ECGPreprocessor` class. Additionally the desired timeframe of the signal can be
specified. We also recommend to define a `RANDOM_STATE` constant to ensure constant behaviour between consecutive runs.
```python
START_TIME = 2251  # [s]
SIGNAL_LENGTH = 300
RANDOM_STATE = 69

DB_PATH = 'data/fantasia_wfdb/f1o01'
ecg_preprocessor = ECGPreprocessor(DB_PATH,
                                   START_TIME,
                                   START_TIME+SIGNAL_LENGTH,
                                   lambda x: savgol_filter(x,
                                            window_length=7,
                                            polyorder=3))

SAMPLING_FREQ = ecg_preprocessor.record.fs
```
Then for further analysis it is necessary to locate the R-peaks timestamps, which can be easily done with the use of the
`ECGPreprocessor` class:
```python
r_timestamps = ecg_preprocessor.get_r_peaks_indexes()/SAMPLING_FREQ
first_qrs = r_timestamps[0]
last_qrs = r_timestamps[-1]
```
The last step of the preprocessing stage is to extract the features of the signal, meaning to cut the ECG signal into 
singular QRS sequences and use them to create a matrix:
```python
qrs_matrix = ecg_preprocessor.create_qrs_matrix(qrs_time_window=.12,
                                                data_centering=True)
```
### EDR extraction
The next stage of the analysis is to apply PCA of ICA algorithms to extraxt the possible candidates for respiratory signal.
 This can be accomplished with the use of the `EDRExtractor` class and it's `pca` and/or `ica` methods.
```python
extractor = EDRExtractor(qrs_matrix)

pca_edr = extractor.pca(random_state=RANDOM_STATE)

ica_edr = extractor.ica(reduce_dim=6, max_iter=160, tol=.1, random_state=RANDOM_STATE)
```
### Postprocessing
The postprocessing stage is the job of the `EDRPicker` class. It is responsible for the candidate signal itnerpolation and 
spectral analysis, whose parameters can be tuned by the user. If one does not want to set any parameters, sensible defaults
 have been put in place.
```python
picker = EDRPicker((first_qrs, last_qrs))
picker.set_spline_params(smoothing=0.1, derivative=0, sampling_frequency=SAMPLING_FREQ)
picker.set_spectral_params(window_width=.04, samples_per_segment=2**13, nfft_=2**16)

edr_ica = picker.apply(ica_edr.T, r_timestamps, method="sugpen", crop_to_freq=.6)
ica_spectra = picker.power_spectra

edr_pca = picker.apply(pca_edr, r_timestamps, method="sugpen", crop_to_freq=.6)
pca_spectra = picker.power_spectra
```
The `apply` method returns the EDR candidates sorted with the desired method specified in the `method` argument. The power
spectra of the candidates can be accessed through the `power_spectra` property.


# Authors
This toolset was created for the "Computational fundamentals of artificial intelligence" project classes at
Gdańsk University of Technology by:
* ***Rafał Osadnik*** [![Rafał's repository](https://gravatar.com/avatar/3d3ddf8d79abc51583f90dbb18c7ee8d?s=15&d=identicon)][rafalosa_repo]
* ***Michał Koruszowic*** [![Michał's repository](https://gravatar.com/avatar/890e077f63151eb8d8feac9cda7585bf?s=15&d=identicon)][mkoruszowic_repo]

# References
This toolset has been developed on the basis of scientific literature.
* *ECG-derived respiration methods: Adapted ICA and PCA*, **Suvi Tiinanen, Kai Noponen, Mikko Tulppo, Antti Kiviniemi,
Tapio Seppänen** https://doi.org/10.1016/j.medengphy.2015.03.004

# Known issues
* Results slightly differ between different operating systems
