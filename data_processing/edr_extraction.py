from sklearn.decomposition import FastICA, PCA
import numpy as np
from typing import Optional


class EDRExtractor:

    """
    Object of the EDRExtractor class extracts the EDR candidates from a given matrix of QRS complexes assigned to the
    object upon its construction.
    """

    def __init__(self, data: np.ndarray):

        self.data: np.ndarray = data

        if data.shape[0] > data.shape[1]:
            self.data = np.reshape(self.data, sorted(data.shape))

    def ica(self, reduce_dim: Optional[int] = None, *args, **kwargs) -> np.ndarray:

        """
        Extract EDR candidates using the independent component analysis.

        :param reduce_dim: Reduce the data dimensionality to the provided number using principal component analysis
        before applying ICA.
        :param args: Arguments for sklearn.decomposition.FastICA.
        :param kwargs: Keyword arguments for sklearn.decomposition.FastICA.
        :return: Numpy array of extracted EDR candidates.
        """

        data = self.data

        if reduce_dim is not None:
            pca = PCA(n_components=reduce_dim,
                      whiten=True,
                      random_state=kwargs["random_state"] if "random_state" in kwargs else None)

            pca.fit(self.data)
            data = pca.components_

        data = data.T

        ica = FastICA(*args, **kwargs)
        return ica.fit_transform(data)*-1

    def pca(self, *args, **kwargs) -> np.ndarray:

        """
        Extract EDR candidates using the principal component analysis.

        :param args: Arguments for sklearn.decomposition.PCA.
        :param kwargs: Keyword arguments for sklearn.decomposition.PCA.
        :return: Numpy array of extracted EDR candidates.
        """

        pca = PCA(*args, **kwargs)
        pca.fit(self.data)
        return pca.components_
