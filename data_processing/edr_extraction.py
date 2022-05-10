from sklearn.decomposition import FastICA, PCA
import numpy as np


# todo: Add pipeline-like functionality.

class EDRExtractor:

    def __init__(self, data: np.ndarray):

        self.data: np.ndarray = data

        if data.shape[0] > data.shape[1]:
            self.data = np.reshape(self.data, sorted(data.shape))

    def ica(self, **kwargs) -> np.ndarray:

        ica = FastICA(**kwargs)
        return ica.fit_transform(self.data.T)

    def pca(self, **kwargs) -> np.ndarray:

        pca = PCA(**kwargs)
        pca.fit(self.data)
        return pca.components_
