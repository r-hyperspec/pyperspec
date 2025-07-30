import numpy as np
from scipy.interpolate import interp1d


def is_iqr_outlier(x: np.ndarray, factor=1.5, **kwargs):
    """Find outliers based on inter-quartile range (IQR)"""
    x = np.array(x).reshape((-1,))
    q1, q3 = tuple(np.nanquantile(x, [0.25, 0.75], **kwargs))
    iqr = q3 - q1
    return (x < (q1 - factor * iqr)) | (x > (q3 + factor * iqr))


def fillna(y: np.ndarray) -> np.ndarray:
    """
    Fills NaN values in a 1D array using linear interpolation.

    Parameters:
        y : np.ndarray
            The input 1D array to fill NaN values.

    Returns:
        np.ndarray
            The array with NaN values filled using linear interpolation.
    """
    x = np.arange(len(y))
    is_nan = np.isnan(y)

    if np.any(is_nan):
        interpolator = interp1d(
            x=x[~is_nan],
            y=y[~is_nan],
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return interpolator(x)

    return y
