"""Plot spectral arrays with Matplotlib."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

__all__ = ("plot_spectra",)


def plot_spectra(
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot each row of a spectral array as a line on one axis.

    Parameters
    ----------
    y : array-like of shape (n_spectra, n_coordinates)
        Real numeric intensities. Both dimensions must be nonempty. NaN values
        create gaps in the lines; infinite values are rejected.
    x : array-like of shape (n_coordinates,), optional
        Finite real numeric coordinates, used in the supplied order. Defaults
        to integer positions starting at zero. Descending coordinates do not
        automatically invert the displayed axis.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. Existing artists are retained. If omitted, create
        a figure containing one axis.
    **kwargs
        Line properties passed to ``Axes.plot``, such as ``color``,
        ``linewidth``, ``linestyle``, and ``alpha``. Without an explicit color,
        Matplotlib's color cycle is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the axis.
    ax : matplotlib.axes.Axes
        Created or supplied axis.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements are violated.
    TypeError
        If the arrays contain nonnumeric or complex values, or ``ax`` is not
        a Matplotlib axis.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pyspc.plotting import plot_spectra
    >>> fig, ax = plot_spectra(
    ...     [[1, 2, 3], [2, 3, 4]], x=[400, 600, 800], color="black"
    ... )
    >>> _ = ax.set_ylabel("Intensity")
    >>> plt.close(fig)

    Prepare a figure first to control its size or layout:

    >>> fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    >>> returned_fig, returned_ax = plot_spectra([[1, 2, 3]], ax=ax)
    >>> returned_ax is ax
    True
    >>> plt.close(fig)
    """
    # Validate arrays before creating a figure or modifying a supplied axis.
    y_array = np.asarray(y)
    if y_array.ndim != 2:
        raise ValueError("y must be a 2D array.")
    if y_array.shape[0] == 0:
        raise ValueError("y must contain at least one spectrum.")
    if y_array.shape[1] == 0:
        raise ValueError("y must contain at least one coordinate.")
    if y_array.dtype.kind not in "iuf":
        raise TypeError("y must contain real numeric values.")
    if np.isinf(y_array).any():
        raise ValueError("y must not contain infinity.")

    x_array = np.arange(y_array.shape[1]) if x is None else np.asarray(x)
    if x_array.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if len(x_array) != y_array.shape[1]:
        raise ValueError(f"x must have length {y_array.shape[1]}.")
    if x_array.dtype.kind not in "iuf":
        raise TypeError("x must contain real numeric values.")
    if not np.isfinite(x_array).all():
        raise ValueError("x must contain only finite values.")

    # Validate the target before drawing.
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes.")

    # Create an axis only when the caller has not supplied one.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Matplotlib treats columns as lines, while spectral arrays use rows.
    ax.plot(x_array, y_array.T, **kwargs)
    return fig, ax
