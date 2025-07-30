"""Provides helper function related to peaks.

Currently, the main function here is `around_max_peak_fit`
which helps to estimate peak position and height based on fitting
a function aroud maximum value.
Other functions related to peak fitting can be found in scipy.signal.*
"""

from typing import Any, Callable, Union, Optional

import numpy as np
import pandas as pd
import scipy


def gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    r"""Gaussian function calculated as:

    .. math::
        y = a\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)

    Notes
    -----
    Do not confuse with probability density function of the normal distribution.
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def lorentzian(x: np.ndarray, a: float, x0: float, gamma: float) -> np.ndarray:
    r"""Lorentzian function calculated as:

    .. math::
        y = a\frac{\gamma^2}{(x-x_0)^2 + gamma^2}
    """
    return a * gamma**2 / ((x - x0) ** 2 + gamma**2)


def parabola_peak_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a peak with a 2nd order polynom

    Parameters
    ----------
    x : ndarray
        1D numeric (vector) of `x` values for the peak fitting
    y : ndarray
        1D numeric (vector) of `y` values for the peak fitting

    Returns
    -------
    dict
        A dictionary with the following keys and values:

        * ``x_max`` : x position of the maximum. Calculated as `-b/2a` from the fitted
            parabola coefficients
        * ``y_max`` : maximum value of the parabola. Calculated as `f(x_max)`
        * ``a``, ``b``, ``c`` : Coefficients of the fitted polynomial `ax^2+bx+c`
    """
    b = np.polyfit(x, y, 2)
    p = np.poly1d(b)
    x_max = -0.5 * b[1] / b[0]
    y_max = p(x_max)
    return {
        "x_max": x_max,
        "y_max": y_max,
        "a": b[0],
        "b": b[1],
        "c": b[2],
    }


def gaussian_peak_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a peak with the Gaussian.

    Parameters
    ----------
    x : np.ndarray
        1D numeric (vector) of `x` values for the peak fitting.
    y : np.ndarray
        1D numeric (vector) of `y` values for the peak fitting.

    Returns
    -------
    dict
        A dictionary with the peak max and the fitting information:

        * ``x_max`` : x position of the maximum.
            Calculated as `x0` from the fitted Gaussian
        * ``y_max`` : maximum value of the parabola.
            Calculated as `a` from the fitted Gaussian (same as `f(x_max)`).
        * ``a``, ``x0``, ``sigma`` : Fitted Gaussian parameters
    """
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))
    popt, pcov = scipy.optimize.curve_fit(gaussian, x, y, p0=[np.max(y), mean, sigma])

    return {
        "x_max": popt[1],
        "y_max": popt[0],
        "a": popt[0],
        "x0": popt[1],
        "sigma": popt[2],
    }


def lorentzian_peak_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a peak with the Lorentzian function

    Parameters
    ----------
    x : np.ndarray
        1D numeric (vector) of `x` values for the peak fitting
    y : np.ndarray
        1D numeric (vector) of `y` values for the peak fitting

    Returns
    -------
    dict:
        a dictionary with the peak max and the fitting infomation:

        * ``x_max``: x position of the maximum.
            Calculated as `x0` from the fitted Lorentzian
        * ``y_max``: maximum value of the parabola.
            Calculated as `a` from the fitted Lorentzian (same as `f(x_max)`)
        * ``a``, ``x0``, ``gamma``: Fitted Lorentzian parameters
    """
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))
    popt, pcov = scipy.optimize.curve_fit(lorentzian, x, y, p0=[np.max(y), mean, sigma])

    return {
        "x_max": popt[1],
        "y_max": popt[0],
        "a": popt[0],
        "x0": popt[1],
        "gamma": popt[2],
    }


def _around_max_peak_fit(
    x: np.ndarray, y: np.ndarray, fit_function: Callable, window: int
) -> Any:
    """Fit a single peak using `window` points around the global maximum value

    First, `x` and `y` vectors are filtered so that only `window` points are
    taken where the middle point corresponds to the global maximum of `y`. Then,
    the fitting function is applied to the filtered values.

    Parameters
    ----------
    x : np.ndarray
        1D numeric (vector) of `x` values for the peak fitting
    y : np.ndarray
        1D numeric (vector) of `y` values for the peak fitting
    fit_function : Callable
        fitting function
    window : int
        window size. Must be and odd number (3,5,7,...)

    Returns
    -------
    Any
        same as output of `fit_function`

    Raises
    ------
    ValueError
        Window size is not odd in the range from 3 to `len(x)`
    """
    # Prepare window size
    if (window >= 3) and (window <= len(x)) and (window % 2):
        window = window // 2
    else:
        raise ValueError(
            "The window size must me an odd number in range from 3 to `len(x)`."
        )

    i_max = np.argmax(y)
    left = max(0, i_max - window)
    right = min(len(y), i_max + window + 1)

    # Test unimodality in the selected region
    # d1_sign = np.sign(np.diff(y[left:right]))
    # d1_sign = d1_sign[d1_sign != 0]
    # if np.sum(np.diff(d1_sign) == -2) != 1:
    #     raise AssertionError(f"y values seem to be not unimodal: {y[left:right]}")

    return fit_function(x[left:right], y[left:right])


METHODS_MAPPING = {
    "parabola": parabola_peak_fit,
    "gaussian": gaussian_peak_fit,
    "lorentzian": lorentzian_peak_fit,
}


def around_max_peak_fit(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: Union[str, Callable] = "max",
    window: Optional[int] = None,
) -> pd.DataFrame:
    """Fit a  peak using `window` points around the global maximum value

    First, `x` and `y` vectors are filtered so that only `window` points are
    taken where the middle point corresponds to the global maximum of `y`. Then,
    the fitting function is applied to the filtered values.

    Parameters
    ----------
    x : np.ndarray
        1D numeric (vector) of `x` values for the peak fitting
    y : np.ndarray
        Either 1D numeric (vector) or a 2D matrix of `y` values for the peak fitting.
        If 2D matrix is given then signals are assumed to be in rows.
    fit_func : str | Callable
        Fitting function. Must be one of:

        * 'max' (default) - returns location of maximum position and value
        * 'parabola', 'gaussian', 'lorentzian' - uses corresponding fit function
        * callable - a function with signature `fun(x, y) -> Iterable`,
          where `x` and `y` are filtered (i.e. `window` around max) values
    window : int
        window size. Must be and odd number (3,5,7,...)

    Returns
    -------
    pd.DataFrame
        A dataframe with the same number of rows as `y` (1D vector considered as
          2D matrix of 1 row ). Each row corresponds to the output of the fitting
          function. In case of string `fit_func`, the dataframe starts with two colums:
          "x_max" (location of fitted peak maximum),
          "y_max" (fitted peak maximum value).
        The following columns correspond to the estimated function parameters.

    Raises
    ------
    ValueError
        Unknown peak fitting method
    """

    # Explicitly convert to numpy array
    x = np.array(x)
    y = np.array(y)

    # Simplify 1D array output
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # window
    if window is None:
        n = len(x)
        window = n if n % 2 else n - 1

    # Parse fit function
    if isinstance(fit_func, str) and (fit_func != "max"):
        fit_func = fit_func.lower()
        if fit_func in METHODS_MAPPING:
            fit_func = METHODS_MAPPING[fit_func]
        else:
            raise ValueError(
                f"Unknown peak fitting method '{fit_func}'. "
                "Must be one of: 'max', 'parabola', 'gaussian', 'lorentzian', "
                "or callable."
            )

    # If method is 'max' then just use faster calculations
    if fit_func == "max":
        result = pd.DataFrame(
            {
                "x_max": x[np.argmax(y, axis=1)],
                "y_max": np.max(y, axis=1),
            }
        )
    elif callable(fit_func):
        resut = [
            _around_max_peak_fit(x, row, fit_function=fit_func, window=window)
            for row in y
        ]
        result = pd.DataFrame.from_records(resut)
    else:
        raise ValueError(f"Unknown peak fitting method '{fit_func}'")

    return result
