from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from skfeature.function.information_theoretical_based.FCBF import fcbf
import numpy as np


class FCBFSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        mode,
        kbest=2,
        threshold=0.0,
        n_bins=2,
    ):
        self.threshold = threshold
        self.selected_features_ = None
        self.n_bins = n_bins
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        self.continuous_idx = None
        self.mode = mode
        self.kbest = kbest

    def fit(self, X, y):
        X = X.copy()

        # Check for continuous features
        self.continuous_idx = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > self.n_bins]

        # Discretize continuous features
        if len(self.continuous_idx) > 0:
            X[:, self.continuous_idx] = self.discretizer.fit_transform(X[:, self.continuous_idx])

        selected_idX = fcbf(X, y, mode=self.mode, delta=self.threshold)  # Returns the list with the features ordered by importance

        # Si se usa "rank", seleccionar k mejores si kbest no es None
        if self.mode == "rank" and self.kbest is not None:

            # Ordenar índices por score y seleccionar los k mejores
            sorted_idx = np.argsort(selected_idX)[::-1][: self.kbest]
            self.selected_features_ = np.array(sorted_idx)
        else:
            self.selected_features_ = np.array(selected_idX)

        return self
