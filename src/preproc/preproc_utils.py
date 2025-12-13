from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from skfeature.function.information_theoretical_based.FCBF import fcbf
import numpy as np
import cupy as cp


def _coerce_y_for_fcbf(y):
    """
    FCBF necesita y discreta (hashable). Si y es Surv (structured),
    usamos solo el campo 'event' como workaround.
    """
    # Caso Surv de sksurv: dtype.names suele ser ('event', 'time')
    if hasattr(y, "dtype") and getattr(y.dtype, "names", None):
        names = set(y.dtype.names)
        if "event" in names:
            return np.asarray(y["event"]).astype(int)
    # DataFrame/Series/array normal
    return np.asarray(y).astype(int).ravel()


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
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf")
        self.continuous_idx = None
        self.mode = mode
        self.kbest = kbest

    def fit(self, X, y):
        X = np.asarray(X).copy()
        y = _coerce_y_for_fcbf(y)  # <-- AÑADIDO

        self.continuous_idx = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > self.n_bins]

        if len(self.continuous_idx) > 0:
            X[:, self.continuous_idx] = self.discretizer.fit_transform(X[:, self.continuous_idx])

        selected_idX = fcbf(X, y, mode=self.mode, delta=self.threshold)

        if self.mode == "rank" and self.kbest is not None:
            # OJO: aquí hay un posible bug lógico (ver nota abajo)
            sorted_idx = np.argsort(selected_idX)[::-1][: self.kbest]
            self.selected_features_ = np.array(sorted_idx)
        else:
            self.selected_features_ = np.array(selected_idX)

        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("FCBFSelector must be fitted before calling transform.")

        X = np.asarray(X).copy()
        if len(self.continuous_idx) > 0:
            X[:, self.continuous_idx] = self.discretizer.transform(X[:, self.continuous_idx])

        return X[:, self.selected_features_]


class ToCuPy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Este paso no hace nada, solo es necesario para cumplir con la API de sklearn
        return self

    def transform(self, X):
        # Convertir la matriz de datos a CuPy
        return cp.array(X)
