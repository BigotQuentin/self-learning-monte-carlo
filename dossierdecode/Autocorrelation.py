import numpy as np
from scipy.optimize import curve_fit

def autocorrelation(x: np.ndarray, t_max: int) -> np.ndarray:
    """Normalized autocorrelation function C(t) / C(0) of the time series x, for t = 0, ..., t_max - 1."""
    x = x - x.mean()
    n = len(x)
    corr = np.array([np.dot(x[: n - t], x[t:]) / (n - t) for t in range(t_max)])
    return corr / corr[0]

def autocorrelation_time(corr: np.ndarray, threshold: float = 0.1) -> float:
    """Fit exp(-t / tau) to the autocorrelation function, as long as it stays above `threshold`."""
    below = np.flatnonzero(corr < threshold)
    n_fit = max(below[0] if len(below) else len(corr), 3)
    t = np.arange(n_fit)
    (tau,), _ = curve_fit(lambda t, tau: np.exp(-t / tau), t, corr[:n_fit], p0=(1.0,))
    return tau