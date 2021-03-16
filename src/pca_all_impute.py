import numpy as np
from bpca import BPCA


class PCAImputer:

    def __init__(self, n_dimension=0):
        self._q = n_dimension
        self._pca = BPCA()

    def fit_transform(self, data, ppca_method='eig', probabilistic=False, n_iteration=100, \
                      verbose=False, print_every=50, trace_mse=False, cdata=None):
        self._data = data.copy()
        self._missing = np.isnan(data)
        self._observed = ~self._missing
        self._mse = np.zeros(n_iteration)

        #         row_defau = np.zeros(self._data.shape[1])
        #         row_means = np.tile(np.nanmean(self._data, axis=0, out=row_defau), [self._data.shape[0], 1])

        # Changed the implemenation here, original below.
        row_defau = np.zeros(self._data.shape[0])
        row_means = np.repeat(np.nanmean(self._data, axis=1, out=row_defau).reshape(-1, 1), \
                              self._data.shape[1], axis=1)
        self._data[self._missing] = row_means[self._missing]
        self._data = np.nan_to_num(self._data)

        for i in range(n_iteration):
            if verbose:
                print("Fitting PCA")
            self._pca.fit(self._data)
            if verbose:
                print("Inverse Transforming")
            self._data[self._missing] = self._pca.inverse_transform(self._pca.transform(self._data))[self._missing]
            if cdata is not None:
                self._mse[i] = np.sum((cdata - self._data) ** 2) / cdata.shape[0]
            if verbose and i % print_every == 0:
                print('Iter %d, MSE=%f' % (i, self._mse[i]))
            if np.abs(self._mse[i - 1] - self._mse[i]) < 1e-6:
                break
        return self._data, self._mse
