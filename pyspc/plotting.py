"""Plot spectral arrays with Matplotlib."""

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, is_color_like, to_rgba_array
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

__all__ = ("plot_spectra",)


def plot_spectra(
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    color: ArrayLike | None = None,
    palette: ArrayLike | Colormap | None = None,
    ymin: ArrayLike | None = None,
    ymax: ArrayLike | None = None,
    fill_kws: Mapping[str, Any] | None = None,
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
    color : Matplotlib color, optional
        One literal color for all spectra, including RGB/RGBA values. The
        Matplotlib alias ``c`` is also accepted, but not with ``color``.
        Cannot be used with ``palette``.
    palette : str, matplotlib.colors.Colormap, or sequence of colors, optional
        Colors assigned in spectrum order. A colormap name or object is
        sampled evenly from 0 to 1. Explicit color sequences are cycled or
        truncated to match the number of spectra. Bands inherit their line's
        color. If neither ``color`` nor ``palette`` is supplied, use the axis's
        color cycle.
    ymin, ymax : array-like of shape (n_spectra, n_coordinates), optional
        Absolute lower and upper bounds of one shaded band per spectrum.
        Supply both or neither. Bounds must be real numeric values, with
        ``ymin <= ymax`` wherever both are present. NaN bounds create gaps;
        infinite bounds are rejected. Bounds define the filled area and are
        not drawn as lines. The spectrum need not lie inside its band.
    fill_kws : mapping, optional
        Band properties passed to ``Axes.fill_between`` independently of line
        properties. By default, each band uses its line's color, alpha 0.3,
        zero outline width, and a z-order one below the line. Requires
        ``ymin`` and ``ymax``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. Existing artists are retained. If omitted, create
        a figure containing one axis.
    **kwargs
        Other line properties passed to ``Axes.plot``, such as ``linewidth``,
        ``linestyle``, and ``alpha``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the axis.
    ax : matplotlib.axes.Axes
        Created or supplied axis.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements are violated, colors or
        palettes are invalid, or both ``color`` and ``palette`` are supplied.
    TypeError
        If the arrays contain nonnumeric or complex values, or ``ax`` is not
        a Matplotlib axis, or both ``color`` and ``c`` are supplied.

    Examples
    --------
    Use one color for all spectra and their filled bands:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from pyspc.plotting import plot_spectra
    >>> y = np.array([[2, 3, 2], [3, 4, 3]])
    >>> x = [400, 600, 800]
    >>> fig, ax = plot_spectra(
    ...     y, x=x, ymin=y - 1, ymax=y + 2, color="black"
    ... )
    >>> plt.close(fig)

    Pass a palette of specific colors. Each band inherits its spectrum's color:

    >>> fig, ax = plot_spectra(
    ...     y, x=x, ymin=y - 1, ymax=y + 2, palette=["tab:blue", "tab:orange"]
    ... )
    >>> plt.close(fig)

    Use a Matplotlib colormap name as the palette:

    >>> fig, ax = plot_spectra(
    ...     y, x=x, ymin=y - 1, ymax=y + 2, palette="viridis"
    ... )
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

    # Validate both bounds before adding any artists.
    if (ymin is None) != (ymax is None):
        raise ValueError("ymin and ymax must be supplied together.")
    bounds = None if ymin is None else (np.asarray(ymin), np.asarray(ymax))
    if bounds is not None:
        for name, bound in zip(("ymin", "ymax"), bounds):
            if bound.shape != y_array.shape:
                raise ValueError(f"{name} must have the same shape as y.")
            if bound.dtype.kind not in "iuf":
                raise TypeError(f"{name} must contain real numeric values.")
            if np.isinf(bound).any():
                raise ValueError(f"{name} must not contain infinity.")
        if np.any(bounds[0] > bounds[1]):
            raise ValueError("ymin must not exceed ymax.")
    elif fill_kws is not None:
        raise ValueError("fill_kws requires band bounds.")
    fill_options = {} if fill_kws is None else dict(fill_kws)

    # Resolve colors before creating a figure or drawing any lines.
    if "c" in kwargs:
        if color is not None:
            raise TypeError("Use only one of color and c.")
        color = kwargs.pop("c")
    line_colors = None
    if color is not None and palette is not None:
        raise ValueError("color and palette cannot be used together.")
    if color is not None:
        if not is_color_like(color):
            raise ValueError("color must be a single Matplotlib color; use palette.")
        kwargs["color"] = color
    elif palette is not None:
        if isinstance(palette, (str, Colormap)):
            cmap = plt.get_cmap(palette) if isinstance(palette, str) else palette
            line_colors = cmap(np.linspace(0, 1, y_array.shape[0]))
        else:
            palette_colors = to_rgba_array(palette)
            if len(palette_colors) == 0:
                raise ValueError("palette must contain at least one color.")
            line_colors = palette_colors[
                np.arange(y_array.shape[0]) % len(palette_colors)
            ]
        kwargs["color"] = line_colors[0]

    # Validate the target before drawing.
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes.")

    # Create an axis only when the caller has not supplied one.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Matplotlib treats columns as lines, while spectral arrays use rows.
    lines = ax.plot(x_array, y_array.T, **kwargs)
    if line_colors is not None:
        for line, line_color in zip(lines, line_colors):
            line.set_color(line_color)

    # Use the resolved line colors without advancing the axis's color cycle.
    if bounds is not None:
        for line, lower, upper in zip(lines, *bounds):
            band_options = dict(fill_options)
            band_options.setdefault("alpha", 0.3)
            band_options.setdefault("zorder", line.get_zorder() - 1)
            if not any(
                key in band_options for key in ("linewidth", "linewidths", "lw")
            ):
                band_options["linewidth"] = 0
            if not any(
                key in band_options
                for key in ("color", "facecolor", "facecolors", "fc")
            ):
                band_options["facecolor"] = line.get_color()
            ax.fill_between(x_array, lower, upper, **band_options)
    return fig, ax
