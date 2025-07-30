import numpy as np

# from scipy import stats, signal

from .utils import is_iqr_outlier, fillna

__all__ = ["find_spikes"]

# def spikes_find_ryabchykov(y: np.array):
#     deltaS = np.diff(y, n=2, axis=1, prepend=y[:, [1]], append=y[:, [-2]])
#     deltaS = np.abs(deltaS)
#     j_max = np.argmax(deltaS, axis=1)
#
#     indices = np.arange(deltaS.shape[1])
#     r = np.array(
#         [
#             np.std(deltaS[i, np.abs(indices - j_max[i]) < 5]) / deltaS[i, j_max[i]]
#             for i in range(len(deltaS))
#         ]
#     )
#     kde = stats.gaussian_kde(r)
#     x_kde = np.linspace(0, 1, 500)
#     y_kde = kde(x_kde)
#     x_max = np.argmax(y_kde)
#     x_kde = x_kde[:x_max]
#     y_kde = y_kde[:x_max]
#
#     _, peaks = signal.find_peaks(y_kde, prominence=0)
#     i_threshold = peaks["right_bases"][np.argmax(peaks["prominences"])]
#
#     is_spiky = np.zeros(y.shape).astype(bool)
#     spiky_rows = np.where(r < x_kde[i_threshold])


def _span_spikes(y: np.ndarray, is_spike: np.ndarray, w: int = 5) -> np.ndarray:
    indices = np.where(is_spike)[0]
    #    /\               *
    #   *  \   ---->     /  \
    #  /    *           /    \
    # /      \         /      \
    max_indices = set()
    for i in indices:
        left = max(i - w, 0)
        right = min(i + w, len(y) - 1)
        max_indices.add(left + np.argmax(y[left:right]))

    #     *                  *
    #    /  \    ---->      *  *
    #   /    \             *    *
    # \/      \/         \*      */
    new_spike_indices = set()
    for i in max_indices:
        left = max(i - 1, 0)
        while (left > 0) and (left > i - w) and (y[left - 1] <= y[left]):
            left -= 1
        right = min(i + 1, len(y) - 1)
        while (right < len(y) - 1) and (right < i + w) and (y[right + 1] <= y[right]):
            right += 1
        new_spike_indices.update(range(left, right + 1))

    new_is_spike = np.zeros(is_spike.shape).astype(bool)
    new_is_spike[np.array(list(new_spike_indices)).astype(int)] = True

    return new_is_spike


def find_spikes(
    y: np.ndarray,
    ndiff: int = 1,
    method: str = "zscore",
    threshold: float = None,
    iqr_factor: float = 7,
):
    """
    Detect spikes in input data based on the specified outlier detection method.

    Parameters
    ----------
    y : np.ndarray
        Input data 1D array of a single spectrum or 2D matrix (spectra in rows)
    ndiff : int, optional
        Order of differentiation, by default 1
    method : str, optional
        Outlier detection method, by default "zscore"
        Available options:
        * 'zscore' for z-score (x[i,j]-mean(x[i,:]))/std(x[i,:])
        * 'mzscore' for modified z-score (x[i,j]-median(x[i,:]))/mad(x[i,:])
        * 'iqr' for inter-quartile range [q1 - factor*iqr, q3 + factor*iqr]
    threshold : float, optional
        Threshold value for outlier/spike detection, by default None.
        Used only with 'zscore' and 'mzscore' methods.
    iqr_factor : float, optional
        Factor for the IQR calculation, by default 7.
        Used only with 'iqr' method.

    Returns
    -------
    np.ndarray
        A boolean matrix indicating the spike locations
    """
    if y.ndim == 1:
        y = y.reshape((1, -1))

    if y.ndim != 2:
        raise ValueError(
            "y expected to be either 2D matrix (spectra in rows) or 1D vector"
        )

    if ndiff == 0:
        data = y
    elif ndiff == 1:
        data = np.diff(y, n=1, axis=1, append=y[:, [-2]])
    elif ndiff == 2:
        data = np.diff(y, n=2, axis=1, prepend=y[:, [1]], append=y[:, [-2]])
    else:
        raise ValueError("Unexpected order of differentiation. Must be one of: 0, 1, 2")

    if method == "zscore":
        mn = np.nanmean(data, axis=1, keepdims=True)
        std = np.nanstd(data, axis=1, keepdims=True)
        is_spiky_mat = np.abs((data - mn) / std) > threshold
    elif method == "mzscore":
        med = np.nanmedian(data, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(data - med), axis=1, keepdims=True)
        # The multiplier 0.6745 is the 0.75th quartile of the standard normal
        # distribution, to which the MAD converges to.
        is_spiky_mat = np.abs(0.6745 * (data - med) / mad) > threshold
    elif method == "iqr":
        is_spiky_mat = np.apply_along_axis(
            lambda x: is_iqr_outlier(x, factor=iqr_factor), axis=1, arr=data
        )
    else:
        raise ValueError(
            "Unexpected spike detection method. Supported options: zscore, mzscore, iqr"
        )

    is_spiky_row = np.any(is_spiky_mat, axis=1)

    if np.any(is_spiky_row):
        # Span spikes
        for i in np.where(is_spiky_row)[0]:
            is_spiky_mat[i, :] = _span_spikes(y[i, :], is_spiky_mat[i, :])

        # Recursively keep searching untill all spikes are cleaned
        spiky_y = y[is_spiky_row, :].copy()
        spiky_y[is_spiky_mat[is_spiky_row, :]] = np.nan
        spiky_y = np.apply_along_axis(fillna, axis=1, arr=spiky_y)

        is_spiky_mat[is_spiky_row, :] = np.bitwise_or(
            is_spiky_mat[is_spiky_row, :],
            find_spikes(
                spiky_y,
                ndiff=ndiff,
                method=method,
                threshold=threshold,
                iqr_factor=iqr_factor,
            ),
        )

    return is_spiky_mat
