"""Plot spectral arrays with Matplotlib."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, is_color_like, to_rgba, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

__all__ = ("plot_spectra",)


def _palette_colors(palette: ArrayLike | Colormap, count: int) -> np.ndarray:
    """Sample a colormap or cycle explicit colors to the requested count.

    Generate ``count`` colors given either list of specific colors
    (e.g. ``["red", "blue"]``) or a cmap name/object (e.g. ``"viridis"``).
    For a list of specific values, colors are generated in cycle.

    Parameters
    ----------
    palette : str, matplotlib.colors.Colormap, or sequence of colors
        Colormap name, colormap object, or explicit colors.
    count : int
        Number of colors to produce. Zero produces an empty color array.

    Returns
    -------
    numpy.ndarray of shape (count, 4)
        RGBA colors in palette order.

    Examples
    --------
    >>> from matplotlib.colors import to_hex
    >>> [to_hex(c) for c in _palette_colors(["red", "blue"], 3)]
    ['#ff0000', '#0000ff', '#ff0000']
    >>> colors = _palette_colors("viridis", 4)
    >>> colors = _palette_colors(plt.get_cmap("viridis"), 4)
    """
    if isinstance(palette, (str, Colormap)):
        cmap = plt.get_cmap(palette) if isinstance(palette, str) else palette
        return cmap(np.linspace(0, 1, count))

    colors = to_rgba_array(palette)
    if len(colors) == 0:
        raise ValueError("palette must contain at least one color.")
    return colors[np.arange(count) % len(colors)]


def _validate_array(
    values: ArrayLike,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    na: Literal["ignore", "raise"] = "ignore",
    inf: Literal["ignore", "raise"] = "raise",
) -> np.ndarray:
    """Convert to a real numeric array and validate its shape and missing values.

    Parameters
    ----------
    values : array-like
        Real numeric values to validate.
    name : str
        Parameter name included in error messages.
    ndim : int, optional
        Required number of dimensions.
    shape : tuple of int, optional
        Required exact shape.
    na : {"ignore", "raise"}, default="ignore"
        Whether NaN values are accepted or rejected.
    inf : {"ignore", "raise"}, default="raise"
        Whether infinite values are accepted or rejected.

    Returns
    -------
    numpy.ndarray
        Validated array, without copying an existing array unnecessarily.

    Raises
    ------
    ValueError
        If shape or missing-value requirements are violated.
    TypeError
        If values are not real numeric data.
    """
    # Check structure before applying numeric value checks.
    array = np.asarray(values)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    # Coordinates reject all nonfinite values; intensities and bounds allow gaps.
    if na == "raise" and np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN.")
    if inf == "raise" and np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinity.")
    return array


def _resolve_color_params(
    color: ArrayLike | None = None,
    hue: ArrayLike | None = None,
    palette: ArrayLike | Colormap | Mapping[Any, Any] | None = None,
    *,
    count: int,
    color_type: Literal["auto", "discrete"] = "auto",
) -> tuple[
    list[ArrayLike], np.ndarray | None, list[Any], str | None, ScalarMappable | None
]:
    """Resolve colors and hue metadata without creating plot artists.

    Parameters
    ----------
    color : Matplotlib color or sequence of colors, optional
        One literal color for all spectra, or exactly one color per spectrum.
        RGB/RGBA vectors represent one color. Cannot be combined with hue or
        palette.
    hue : array-like of shape (count,), optional
        Numeric values or categories used to map colors to spectra.
    palette : str, colormap, sequence of colors, or mapping, optional
        Color source for spectra or hue categories. Mappings require hue.
    count : int
        Number of spectra whose colors must be resolved.
    color_type : {"auto", "discrete"}, default="auto"
        Map numeric hue continuously unless categorical or explicitly discrete.

    Returns
    -------
    line_colors : list
        One Matplotlib color per spectrum. Defaults to the RGBA value of C0
        from the current style for every spectrum.
    codes : numpy.ndarray or None
        Per-spectrum category codes, or None without discrete hue.
    categories : list
        Category labels in legend order; empty without discrete hue.
    title : str or None
        Hue Series name, if present.
    mappable : matplotlib.cm.ScalarMappable or None
        Shared normalization and colormap for continuous hue. None for discrete
        hue, absent hue, or numeric hue with no observed values.

    Examples
    --------
    Unspecified colors share the first style color. A literal selects another:

    >>> colors, codes, categories, title, mappable = _resolve_color_params(count=2)
    >>> colors, codes, categories, title, mappable = _resolve_color_params(
    ...     color="black", count=2
    ... )
    >>> colors
    ['black', 'black']

    Hue codes follow pandas category order, with -1 for missing values:

    >>> groups = pd.Series(["B", "A", "B", None], name="Group")
    >>> colors, codes, categories, title, mappable = _resolve_color_params(
    ...     hue=groups, count=4
    ... )
    >>> codes.tolist(), categories, title
    ([1, 0, 1, -1], ['A', 'B'], 'Group')

    A categorical input preserves its declared order:

    >>> groups = pd.Categorical(["B", "A"], categories=["B", "A", "unused"])
    >>> colors, codes, categories, title, mappable = _resolve_color_params(
    ...     hue=groups, count=2
    ... )
    >>> codes.tolist(), categories
    ([0, 1], ['B', 'A'])

    Numeric hue resolves to a continuous scale:

    >>> colors, codes, categories, title, mappable = _resolve_color_params(
    ...     hue=[10, 11, 20], palette="viridis", count=3
    ... )
    >>> mappable.norm([10, 11, 20]).tolist()
    [0.0, 0.1, 1.0]
    """
    # Reject competing instructions before selecting a color source.
    if color_type not in ("auto", "discrete"):
        raise ValueError("color_type must be 'auto' or 'discrete'.")
    if color is not None and palette is not None:
        raise ValueError("color and palette cannot be used together.")
    if color is not None and hue is not None:
        raise ValueError("color and hue cannot be used together.")
    if hue is None and palette is not None and isinstance(palette, Mapping):
        raise ValueError("palette mappings require hue.")
    if hue is not None and (isinstance(hue, (str, bytes)) or np.ndim(hue) != 1):
        raise ValueError("hue must be a 1D vector with one value per spectrum.")
    if hue is not None and len(hue) != count:
        raise ValueError(f"hue must have length {count}.")

    # If nothing is provided we just give a single value
    if hue is None and palette is None and color is None:
        return [to_rgba("C0")] * count, None, [], None, None

    # Preserve categorical inputs; infer ordinary numeric values without parsing strings
    if hue is not None:
        values = pd.Series(hue).reset_index(drop=True)
        title = None if values.name is None else str(values.name)
        hue_dtype = values.dropna().infer_objects().dtype
        is_continuous = (
            color_type == "auto"
            and not isinstance(values.dtype, pd.CategoricalDtype)
            and pd.api.types.is_numeric_dtype(hue_dtype)
            and not pd.api.types.is_bool_dtype(hue_dtype)
        )

        # Resolve finite numeric hue and its scale before creating any artists.
        if is_continuous:
            if pd.api.types.is_complex_dtype(hue_dtype):
                raise TypeError("hue must contain real numeric values.")
            if palette is not None and not isinstance(palette, (str, Colormap)):
                raise ValueError(
                    "Continuous hue requires a colormap name or object; "
                    "use color_type='discrete' for color sequences or mappings."
                )
            hue_array = _validate_array(
                values.astype("Float64").to_numpy(dtype=float, na_value=np.nan),
                "hue",
                ndim=1,
            )
            cmap = plt.get_cmap("viridis" if palette is None else palette)
            present = ~np.isnan(hue_array)
            resolved_colors = np.tile(to_rgba("gray"), (count, 1))
            if not present.any():
                return list(resolved_colors), None, [], title, None
            observed = hue_array[present]
            # Keep constant and nearly constant scales centered and nonsingular.
            vmin, vmax = float(observed.min()), float(observed.max())
            tolerance = max(
                max(abs(vmin), abs(vmax)) * 1e-12, np.finfo(float).tiny * 1e21
            )
            if vmax - vmin <= tolerance:
                padding = max(abs(vmin) * 0.1, abs(vmax) * 0.1, 0.1)
                vmin, vmax = vmin - padding, vmax + padding
            if not np.isfinite(vmax - vmin):
                raise ValueError("hue range is too large for numeric normalization.")
            mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
            mappable.set_array(observed)
            resolved_colors[present] = mappable.to_rgba(observed)
            return list(resolved_colors), None, [], title, mappable

        # Resolve discrete codes and keep missing values separate from category labels.
        values = values.astype("category")
        values = values.cat.remove_unused_categories()
        codes = values.cat.codes.to_numpy()
        categories = values.cat.categories.tolist()

        # From hue values to specific RGB colors
        if palette is None:
            hue_palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ("C0",))
            category_colors = _palette_colors(hue_palette, len(categories))
        elif isinstance(palette, Mapping):
            if not set(categories).issubset(palette.keys()):
                missing = list(set(categories).difference(palette.keys()))
                raise ValueError(f"palette is missing hue categories: {missing!r}.")
            category_colors = to_rgba_array(
                [palette[category] for category in categories]
            )
        else:
            category_colors = _palette_colors(palette, len(categories))

        resolved_colors = np.tile(to_rgba("gray"), (count, 1))
        present = codes >= 0
        resolved_colors[present] = category_colors[codes[present]]
        return list(resolved_colors), codes, categories, title, None

    # Palette provided without hue
    if palette is not None:
        return _palette_colors(palette, count).tolist(), None, [], None, None

    # Specific colors are provided
    if is_color_like(color):
        return [color] * count, None, [], None, None
    try:
        line_colors = to_rgba_array(color)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "color must be a Matplotlib color or a sequence of colors."
        ) from exc
    if len(line_colors) != count:
        raise ValueError(f"color must contain exactly {count} colors.")
    return list(line_colors), None, [], None, None


def _add_legend(
    ax: Axes,
    lines: Sequence[Line2D],
    codes: np.ndarray,
    categories: Sequence[Any],
    title: str | None,
) -> None:
    """Add one entry per observed category, followed by missing values if present."""
    first_index_by_code = {}
    for index, code in enumerate(codes):
        first_index_by_code.setdefault(code, index)
    handles = [lines[first_index_by_code[code]] for code in range(len(categories))]
    labels = [str(category) for category in categories]
    if -1 in first_index_by_code:
        handles.append(lines[first_index_by_code[-1]])
        labels.append("Missing")
    ax.legend(handles, labels, title=title)


def plot_spectra(
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    hue: ArrayLike | None = None,
    color: ArrayLike | None = None,
    palette: ArrayLike | Colormap | Mapping[Any, Any] | None = None,
    color_type: Literal["auto", "discrete"] = "auto",
    legend: bool = True,
    ymin: ArrayLike | None = None,
    ymax: ArrayLike | None = None,
    fill_kws: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot spectra with optional hue colors and shaded bands on one axis.

    Parameters
    ----------
    y : array-like of shape (n_spectra, n_coordinates)
        Real numeric intensities. Both dimensions must be nonempty. NaN values
        create gaps in the lines; infinite values are rejected.
    x : array-like of shape (n_coordinates,), optional
        Finite real numeric coordinates, used in the supplied order. Defaults
        to integer positions starting at zero. Descending coordinates do not
        automatically invert the displayed axis.
    hue : array-like of shape (n_spectra,), optional
        Values used to assign colors, one value per spectrum. Numeric values
        use a continuous color scale by default. Booleans and
        pandas categorical inputs use discrete colors and a legend.
        Discrete category order follows pandas' conversion (sorted when
        possible). Existing categorical inputs retain their declared order.
        Unused categories are removed.
        Series indices are ignored; values align with spectral rows by position.
        Missing values are drawn gray; discrete legends label them "Missing".
        Continuous hue must be finite except for missing values. All-missing
        numeric hue draws gray spectra. Constant numeric hue uses the colormap
        midpoint. Constant and nearly constant ranges are expanded to keep a
        stable color scale.
        Ranges too large for floating-point normalization are rejected.
        A named Series labels the categorical legend. Spectra are not aggregated.
    color : Matplotlib color or sequence of colors, optional
        One literal color for all spectra, including RGB/RGBA values, or a
        sequence containing exactly one color per spectrum. A numeric RGB/RGBA
        vector is interpreted as a single color even if its length matches the
        spectrum count. Use a sequence of RGB/RGBA vectors for individual colors.
        The Matplotlib alias ``c`` is also accepted, but not with ``color``.
        Cannot be used with ``palette`` or ``hue``.
    palette : str, colormap, sequence of colors, or mapping, optional
        For continuous hue, a colormap name or object, defaulting to "viridis".
        Values map linearly over the observed numeric range.
        For discrete hue, colors assigned to categories, or in spectrum order
        without hue. In these cases, a colormap name or object is sampled
        evenly from 0 to 1. Explicit color sequences are cycled or truncated
        to match the category or spectrum count. A mapping assigns colors by
        category and requires hue;
        it must cover all observed nonmissing categories. Extra keys are allowed.
        With discrete hue, the default palette is Matplotlib's style color cycle.
        With no color, hue, or palette, every spectrum uses Matplotlib C0,
        the first color in the configured style.
        Bands inherit their line's color.
    color_type : {"auto", "discrete"}, default="auto"
        Automatic continuous mapping for numeric hue, or discrete mapping for
        all hue values. Use "discrete" for numeric sample IDs or treatment codes.
    legend : bool, default=True
        Draw a legend for discrete hue, with one entry per category and a
        missing-value entry when present. Continuous hue currently draws no
        colorbar. If False or hue is omitted, existing legends are retained.
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
        palettes or hue are invalid, or color is combined with palette or hue.
    TypeError
        If the arrays contain nonnumeric or complex values, or ``ax`` is not
        a Matplotlib axis, hue values are unhashable, or both ``color`` and
        ``c`` are supplied.

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

    Pass one specific color per spectrum. Each band inherits its spectrum's color:

    >>> fig, ax = plot_spectra(
    ...     y, x=x, ymin=y - 1, ymax=y + 2, color=["tab:blue", "tab:orange"]
    ... )
    >>> plt.close(fig)

    Use a shorter palette to cycle colors across spectra:

    >>> fig, ax = plot_spectra(y, palette=["tab:blue"])
    >>> plt.close(fig)

    Use a Matplotlib colormap name as the palette:

    >>> fig, ax = plot_spectra(
    ...     y, x=x, ymin=y - 1, ymax=y + 2, palette="viridis"
    ... )
    >>> plt.close(fig)

    Map repeated categories to the same color, with one legend entry per category:

    >>> fig, ax = plot_spectra(
    ...     [[2, 3, 2], [3, 4, 3], [4, 5, 4]], x=x,
    ...     hue=["A", "A", "B"], palette={"A": "tab:blue", "B": "tab:orange"}
    ... )
    >>> plt.close(fig)

    Declare category order with pandas; spectral row order stays unchanged:

    >>> import pandas as pd
    >>> groups = pd.Categorical(["B", "A"], categories=["A", "B"], ordered=True)
    >>> fig, ax = plot_spectra(y, hue=groups, palette=["tab:blue", "tab:orange"])
    >>> plt.close(fig)

    Map numeric values continuously using a colormap:

    >>> dose = pd.Series([10, 20], name="Dose")
    >>> fig, ax = plot_spectra(
    ...     y, hue=dose, palette="plasma", ymin=y - 1, ymax=y + 1
    ... )
    >>> plt.close(fig)

    Treat numeric IDs as categories with specific colors and a legend:

    >>> fig, ax = plot_spectra(
    ...     y, hue=[20, 10], color_type="discrete", palette={10: "red", 20: "blue"}
    ... )
    >>> plt.close(fig)
    """
    # Validate arrays before creating a figure or modifying a supplied axis.
    y_array = _validate_array(y, "y", ndim=2)
    if y_array.shape[0] == 0:
        raise ValueError("y must contain at least one spectrum.")
    if y_array.shape[1] == 0:
        raise ValueError("y must contain at least one coordinate.")
    x_array = (
        np.arange(y_array.shape[1])
        if x is None
        else _validate_array(x, "x", ndim=1, na="raise")
    )
    if len(x_array) != y_array.shape[1]:
        raise ValueError(f"x must have length {y_array.shape[1]}.")

    # Validate both bounds before adding any artists.
    if (ymin is None) != (ymax is None):
        raise ValueError("ymin and ymax must be supplied together.")
    bounds = (
        None
        if ymin is None
        else (
            _validate_array(ymin, "ymin", shape=y_array.shape),
            _validate_array(ymax, "ymax", shape=y_array.shape),
        )
    )
    if bounds is not None:
        if np.any(bounds[0] > bounds[1]):
            raise ValueError("ymin must not exceed ymax.")
    elif fill_kws is not None:
        raise ValueError("fill_kws requires band bounds.")
    fill_options = {} if fill_kws is None else dict(fill_kws)

    # Normalize the Matplotlib alias before resolving all color parameters.
    if "c" in kwargs:
        if color is not None:
            raise TypeError("Use only one of color and c.")
        color = kwargs.pop("c")
    line_colors, hue_codes, categories, hue_title, _mappable = _resolve_color_params(
        color, hue, palette, count=y_array.shape[0], color_type=color_type
    )

    # Validate the target before drawing.
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes.")

    # Create an axis only when the caller has not supplied one.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Matplotlib treats columns as lines, while spectral arrays use rows.
    lines = ax.plot(x_array, y_array.T, color=line_colors[0], **kwargs)
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

    # TODO: Add side colorbars for numeric hue with the multiple-axis layout.
    if hue_codes is not None and legend:
        _add_legend(ax, lines, hue_codes, categories, hue_title)
    return fig, ax
