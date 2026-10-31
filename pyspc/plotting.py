"""Plot spectral arrays with Matplotlib."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from numbers import Integral
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cbook import normalize_kwargs
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, is_color_like, to_rgba, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

from ._plotting_breaks import _validate_spacing, _validate_x_breaks, break_xaxis

__all__ = ("plot_spectra", "break_xaxis")


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


def _validate_spectra_arrays(
    y: ArrayLike,
    x: ArrayLike | None,
    ymin: ArrayLike | None,
    ymax: ArrayLike | None,
    fill_kws: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Validate spectral arrays and band bounds before allocating any axes."""
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
    return y_array, x_array, bounds


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


def _resolve_hue_styles(
    hue_styles: Mapping[Any, Mapping[str, Any]] | None,
    codes: np.ndarray | None,
    categories: Sequence[Any],
) -> list[dict[str, Any]] | None:
    """Validate discrete hue overrides and resolve one style per spectral row."""
    if hue_styles is None:
        return None
    if not isinstance(hue_styles, Mapping):
        raise TypeError("hue_styles must be a mapping of categories to style mappings.")
    if codes is None:
        raise ValueError(
            "hue_styles requires discrete hue; use color_type='discrete' "
            "for numeric categories."
        )

    # Normalize aliases and validate styles without creating axes or changing artists.
    styles_by_code = []
    for category in categories:
        options = hue_styles.get(category, {})
        if not isinstance(options, Mapping):
            raise TypeError("Each hue_styles entry must be a style mapping.")
        options = normalize_kwargs(options, Line2D)
        if set(options).difference(("linewidth", "linestyle", "alpha")):
            raise ValueError(
                "hue_styles supports only linewidth, linestyle, and alpha."
            )
        Line2D([], [], **options)
        styles_by_code.append(options)

    # Missing hue and unspecified categories inherit the global line settings.
    return [{} if code < 0 else styles_by_code[code] for code in codes]


def _add_legend(
    target: Axes | Figure,
    lines: Sequence[Line2D],
    codes: np.ndarray,
    categories: Sequence[Any],
    title: str | None,
    **kwargs: Any,
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
    target.legend(handles, labels, title=title, **kwargs)


def _facet_membership(
    values: ArrayLike | None, name: str, count: int
) -> tuple[np.ndarray, list[Any], str | None]:
    """Return facet codes, raw category values, and the optional Series name."""
    if values is None:
        return np.zeros(count, dtype=int), [None], None
    if isinstance(values, (str, bytes)) or np.ndim(values) != 1:
        raise ValueError(f"{name} must be a 1D vector with one value per spectrum.")
    if len(values) != count:
        raise ValueError(f"{name} must have length {count}.")

    # Keep missing codes separate from literal labels such as "Missing".
    categories = pd.Series(values).astype("category").cat.remove_unused_categories()
    codes = categories.cat.codes.to_numpy().copy()
    category_values = categories.cat.categories.tolist()
    if (codes == -1).any():
        codes[codes == -1] = len(category_values)
        category_values.append(pd.NA)
    series_name = None if categories.name is None else str(categories.name)
    return codes, category_values, series_name


def _default_title_formatter(
    row_value: Any | None,
    col_value: Any | None,
    *,
    row_name: str | None = None,
    col_name: str | None = None,
) -> str:
    """Join row and column labels, including Series names when available."""
    labels = []
    for value, name in ((row_value, row_name), (col_value, col_name)):
        if value is None:
            continue
        label = "Missing" if value is pd.NA else str(value)
        if name is not None:
            label = f"{name} = {label}"
        if label:
            labels.append(label)
    return " | ".join(labels)


def _generate_facet_titles(
    row_values: Sequence[Any | None],
    col_values: Sequence[Any | None],
    title_formatter: Callable[[Any | None, Any | None], str],
) -> list[list[str]]:
    """Call the formatter once per facet and validate each returned title."""
    facet_titles = []
    for row_value in row_values:
        titles = []
        for col_value in col_values:
            title = title_formatter(row_value, col_value)
            if not isinstance(title, str):
                raise TypeError("title_formatter must return a str.")
            titles.append(title)
        facet_titles.append(titles)
    return facet_titles


def plot_spectra(
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    row: ArrayLike | None = None,
    col: ArrayLike | None = None,
    col_wrap: int | None = None,
    title_formatter: Callable[[Any | None, Any | None], str] | None = None,
    sharex: bool | Literal["row", "col"] = True,
    sharey: bool | Literal["row", "col"] = True,
    x_breaks: ArrayLike | None = None,
    x_breaks_gap: float | None = None,
    xtick_major: float | None = None,
    xtick_minor: float | None = None,
    hue: ArrayLike | None = None,
    hue_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    color: ArrayLike | None = None,
    palette: ArrayLike | Colormap | Mapping[Any, Any] | None = None,
    color_type: Literal["auto", "discrete"] = "auto",
    legend: bool = True,
    symmetric: bool = False,
    ymin: ArrayLike | None = None,
    ymax: ArrayLike | None = None,
    fill_kws: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot spectra and optional bands on one axis or a row/column facet grid.

    Parameters
    ----------
    y : array-like of shape (n_spectra, n_coordinates)
        Real numeric intensities. Both dimensions must be nonempty. NaN values
        create gaps in the lines; infinite values are rejected.
    x : array-like of shape (n_coordinates,), optional
        Finite real numeric coordinates, used in the supplied order. Defaults
        to integer positions starting at zero. Descending coordinates do not
        automatically invert the displayed axis.
    row, col : array-like of shape (n_spectra,), optional
        Membership of each spectrum in the facet rows and columns. Values align
        by position; Series indices are ignored. Category order follows pandas,
        preserving declared categorical order and omitting unused levels.
        Missing membership forms a separate facet labeled "Missing", after the
        observed categories. All row-column combinations remain visible,
        including empty panels. Named Series produce titles such as
        "Treatment = A"; unnamed vectors use the category label alone.
        Combined titles join the row and column labels with " | ".
    col_wrap : positive int, optional
        Wrap column facets at this width, filling panels from left to right,
        then top to bottom. Requires col and cannot be combined with row.
        The grid retains the requested width even when there are fewer facets;
        unused axes in the last row are hidden. Category order and missing
        membership follow col as usual.
    title_formatter : callable, optional
        Called as title_formatter(row_value, col_value) to produce a string
        title for each facet, including empty row-column combinations. Receives
        the original category values, with None for an unused dimension and
        pandas.NA for missing membership. Literal category strings such as
        "Missing" are unchanged. Series names are not passed; include desired
        names in the callable itself. Hidden wrap slots are skipped. Used only
        with row or col; unfaceted plots do not call the formatter. If None,
        retain the default titles described above. Formatter exceptions are
        propagated before creating the grid.
    sharex, sharey : bool or {"row", "col"}, default=True
        Share coordinate or intensity axes across all facets, within each row,
        within each column, or not at all. Used only when faceting. With
        col_wrap, rows and columns refer to physical positions in the grid.
    x_breaks : array-like of shape (n_breaks, 2), optional
        X-coordinate intervals to omit, such as [(1800, 2600)]. Each interval
        has lower < upper, even for descending coordinates. Intervals must be
        finite, disjoint, and not touching. Applied after plotting to each
        visible spectral panel; legends and colorbars are not broken. Requires
        a linear x scale. See break_xaxis for limitations on clipping, ticks,
        and normal axis operations after breaking.
    x_breaks_gap : positive float, optional
        Displayed gap width in x-data units, with the same scale as retained
        coordinates. Defaults to xtick_minor when provided, otherwise half the
        major-tick interval (automatic if xtick_major is None). Only used when
        there are visible x_breaks.
    xtick_major, xtick_minor : positive float, optional
        Major and minor tick spacings in original x-data units, anchored at
        zero. Supported with or without breaks. Ticks inside omitted intervals
        are excluded. Explicit major spacing is honored without thinning,
        even if labels become crowded. Without breaks, None leaves the existing
        locator unchanged. With breaks, None selects automatic major ticks or
        disables minor ticks, respectively.
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
        A named Series labels the legend or colorbar. Spectra are not aggregated.
    hue_styles : mapping of categories to style mappings, optional
        Per-category line overrides for linewidth (or lw), linestyle (or ls),
        and alpha. Requires discrete hue; for numeric categories, supply
        color_type="discrete" or a pandas categorical vector. Each override
        takes precedence over the corresponding global line property in kwargs.
        Unspecified groups and properties, including missing hue, retain their
        global settings. Extra category keys are ignored. Colors remain
        controlled by palette; other properties are not accepted here.
        Styles are consistent across facets and reflected in the legend.
        Overrides apply only to lines; use fill_kws to style shaded bands.
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
        Bands inherit their line's color. Colors are resolved across all spectra
        before faceting, so hue scales and per-spectrum colors remain consistent
        across panels.
    color_type : {"auto", "discrete"}, default="auto"
        Automatic continuous mapping for numeric hue, or discrete mapping for
        all hue values. Use "discrete" for numeric sample IDs or treatment codes.
    legend : bool, default=True
        Draw a hue guide: a legend for discrete hue, or a vertical colorbar on
        the right for continuous hue. Discrete legends contain one entry per
        category and a missing-value entry when present. Continuous colorbars
        use the same global scale as the spectra and bands; missing hue stays
        gray and all-missing numeric hue produces no colorbar. A named hue
        Series labels the guide. Facet grids have one shared guide outside the
        panels, with space reserved by Matplotlib's constrained layout.
        Unfaceted calls use an axis legend or a colorbar beside the created or
        supplied axis. If False or hue is omitted, no guide is added; existing
        legends and colorbars are retained.
    symmetric : bool, default=False
        Make y-axis limits symmetric around zero after plotting. The extent is
        the largest absolute value of the axis's current limits, including
        margins, shaded bands, existing artists, or manually configured limits.
        For facets, apply symmetry after all panels are drawn, using the full
        extent of each shared y-axis group.
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
        a figure containing one axis. Cannot be supplied with row or col;
        facet grids create their own figure and axes.
    **kwargs
        Other line properties passed to ``Axes.plot``, such as ``linewidth``,
        ``linestyle``, and ``alpha``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the spectral axes and any colorbar axis.
    ax : matplotlib.axes.Axes or numpy.ndarray of Axes
        Created or supplied axis without facets. With row or col, a 2D array of
        shape (n_rows, n_columns), including when either dimension is one.
        Wrapped grids include hidden axes for unused positions in the last row.
        The colorbar axis is not included in this return value.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements are violated, colors or
        palettes or hue are invalid, color is combined with palette or hue,
        facet membership or sharing options are invalid, or ax is supplied with
        facets, or col_wrap is not a positive integer or is used without col
        or with row, or hue_styles is used without discrete hue or contains
        unsupported properties or invalid style values.
    TypeError
        If the arrays contain nonnumeric or complex values, or ``ax`` is not
        a Matplotlib axis, hue values are unhashable, or both ``color`` and
        ``c`` are supplied, or title_formatter is not callable or returns a
        non-string title, or hue_styles or an observed category's style is not
        a mapping.

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

    Override line properties for selected discrete hue groups:

    >>> fig, ax = plot_spectra(
    ...     y, hue=["Control", "Treated"], linewidth=1,
    ...     hue_styles={"Control": {"linestyle": "--", "alpha": 0.6},
    ...                 "Treated": {"linewidth": 2.5}}
    ... )
    >>> [line.get_linewidth() for line in ax.lines]
    [1.0, 2.5]
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

    Center the y-axis limits on zero while retaining the plotted extent:

    >>> fig, ax = plot_spectra(y, symmetric=True)
    >>> plt.close(fig)

    Split spectra into panels, keeping a consistent hue palette across them:

    >>> treatment = pd.Series(["Control", "Treated"], name="Treatment")
    >>> fig, axes = plot_spectra(y, col=treatment, hue=["A", "B"])
    >>> axes.shape
    (1, 2)
    >>> plt.close(fig)

    Wrap column facets into multiple rows:

    >>> fig, axes = plot_spectra(y, col=treatment, col_wrap=1)
    >>> axes.shape
    (2, 1)
    >>> plt.close(fig)

    Format facet titles using their category values:

    >>> fig, axes = plot_spectra(
    ...     y, col=treatment,
    ...     title_formatter=lambda row_value, col_value: f"Treatment: {col_value}"
    ... )
    >>> axes[0, 0].get_title()
    'Treatment: Control'
    >>> plt.close(fig)

    Omit an interval of spectral coordinates:

    >>> fig, ax = plot_spectra(
    ...     y, x=x, x_breaks=[(500, 700)], x_breaks_gap=50,
    ...     xtick_major=100, xtick_minor=20
    ... )
    >>> plt.close(fig)
    """
    # Reject malformed intervals before plotting or allocating any axes.
    if x_breaks is not None:
        x_breaks = _validate_x_breaks(x_breaks, ax=ax)
    x_breaks_gap = _validate_spacing(x_breaks_gap, "x_breaks_gap")
    xtick_major = _validate_spacing(xtick_major, "xtick_major")
    xtick_minor = _validate_spacing(xtick_minor, "xtick_minor")

    # Validate the callback without calling it on an unfaceted plot.
    if title_formatter is not None and not callable(title_formatter):
        raise TypeError("title_formatter must be callable or None.")

    # Validate wrapping even when no facet membership has been supplied.
    if col_wrap is not None:
        if (
            isinstance(col_wrap, bool)
            or not isinstance(col_wrap, Integral)
            or col_wrap < 1
        ):
            raise ValueError("col_wrap must be a positive integer.")
        if row is not None:
            raise ValueError("col_wrap cannot be combined with row.")
        if col is None:
            raise ValueError("col_wrap requires col.")
        col_wrap = int(col_wrap)

    if row is None and col is None:
        fig, ax = _single_ax_spectra_plot(
            y,
            x=x,
            hue=hue,
            hue_styles=hue_styles,
            color=color,
            palette=palette,
            color_type=color_type,
            legend=legend,
            symmetric=symmetric,
            ymin=ymin,
            ymax=ymax,
            fill_kws=fill_kws,
            ax=ax,
            **kwargs,
        )
        if x_breaks is not None or xtick_major is not None or xtick_minor is not None:
            break_xaxis(
                ax,
                [] if x_breaks is None else x_breaks,
                x_breaks_gap=x_breaks_gap,
                xtick_major=xtick_major,
                xtick_minor=xtick_minor,
            )
        return fig, ax

    # Validate the full dataset and facet membership before creating a grid.
    if ax is not None:
        raise ValueError("ax cannot be supplied with facets.")
    for name, sharing in (("sharex", sharex), ("sharey", sharey)):
        if sharing not in (True, False, "row", "col"):
            raise ValueError(f"{name} must be True, False, 'row', or 'col'.")
    y_array, x_array, bounds = _validate_spectra_arrays(y, x, ymin, ymax, fill_kws)
    count = y_array.shape[0]
    row_codes, row_values, row_name = _facet_membership(row, "row", count)
    col_codes, col_values, col_name = _facet_membership(col, "col", count)

    # Resolve colors globally so subsets never restart a palette or hue scale.
    if "c" in kwargs:
        if color is not None:
            raise TypeError("Use only one of color and c.")
        color = kwargs.pop("c")
    colors, hue_codes, categories, hue_title, mappable = _resolve_color_params(
        color, hue, palette, count=count, color_type=color_type
    )
    colors = to_rgba_array(colors)
    line_styles = _resolve_hue_styles(hue_styles, hue_codes, categories)

    # Format all real facets before creating a figure, including empty panels.
    if title_formatter is None:
        title_formatter = partial(
            _default_title_formatter, row_name=row_name, col_name=col_name
        )
    facet_titles = _generate_facet_titles(row_values, col_values, title_formatter)

    # Retain empty combinations and wrap columns into physical rows when needed.
    ncols = len(col_values) if col_wrap is None else col_wrap
    nrows = (
        len(row_values) if col_wrap is None else (len(col_values) + ncols - 1) // ncols
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        layout="constrained",
    )
    lines_by_index: dict[int, Line2D] = {}
    for (i, j), facet_ax in np.ndenumerate(axes):
        # Map grid positions back to membership before slicing spectral rows.
        facet_row, facet_col = (i, j) if col_wrap is None else (0, i * ncols + j)
        if facet_col >= len(col_values):
            facet_ax.set_visible(False)
            continue
        facet_ax.set_title(facet_titles[facet_row][facet_col])
        indices = np.flatnonzero((row_codes == facet_row) & (col_codes == facet_col))
        if not len(indices):
            continue
        _single_ax_spectra_plot(
            y_array[indices],
            x=x_array,
            color=colors[indices],
            ymin=None if bounds is None else bounds[0][indices],
            ymax=None if bounds is None else bounds[1][indices],
            fill_kws=fill_kws,
            ax=facet_ax,
            **kwargs,
        )
        for index, line in zip(indices, facet_ax.lines):
            if line_styles is not None:
                line.set(**line_styles[index])
            lines_by_index[index] = line

    # A hidden final panel must not suppress the tick labels directly above it.
    if col_wrap is not None and sharex in (True, "col"):
        for j in range(min(ncols, len(col_values))):
            bottom_row = (len(col_values) - 1 - j) // ncols
            axes[bottom_row, j].tick_params(axis="x", labelbottom=True)

    # Draw one shared hue guide using the globally resolved colors and scale.
    if hue_codes is not None and legend:
        _add_legend(
            fig,
            [lines_by_index[index] for index in range(count)],
            hue_codes,
            categories,
            hue_title,
            loc="outside right upper",
        )
    elif mappable is not None and legend:
        fig.colorbar(
            mappable,
            ax=[facet_ax for facet_ax in axes.flat if facet_ax.get_visible()],
            location="right",
            label=hue_title or "",
        )

    # Setting limits disables autoscaling, so wait until all panels are drawn.
    if symmetric:
        for facet_ax in axes.flat:
            if not facet_ax.get_visible():
                continue
            limit = max(abs(value) for value in facet_ax.get_ylim())
            facet_ax.set_ylim(-limit, limit)

    # Configure x axes only on visible spectral panels, after all artists exist.
    if x_breaks is not None or xtick_major is not None or xtick_minor is not None:
        for facet_ax in axes.flat:
            if facet_ax.get_visible():
                break_xaxis(
                    facet_ax,
                    [] if x_breaks is None else x_breaks,
                    x_breaks_gap=x_breaks_gap,
                    xtick_major=xtick_major,
                    xtick_minor=xtick_minor,
                )

    return fig, axes


def _single_ax_spectra_plot(
    y: ArrayLike,
    x: ArrayLike | None = None,
    *,
    hue: ArrayLike | None = None,
    hue_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    color: ArrayLike | None = None,
    palette: ArrayLike | Colormap | Mapping[Any, Any] | None = None,
    color_type: Literal["auto", "discrete"] = "auto",
    legend: bool = True,
    symmetric: bool = False,
    ymin: ArrayLike | None = None,
    ymax: ArrayLike | None = None,
    fill_kws: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Draw spectra and optional bands on a supplied or newly created axis.

    Single-axis parameters and return values match :func:`plot_spectra`.
    """
    y_array, x_array, bounds = _validate_spectra_arrays(y, x, ymin, ymax, fill_kws)
    fill_options = {} if fill_kws is None else dict(fill_kws)

    # Normalize the Matplotlib alias before resolving all color parameters.
    if "c" in kwargs:
        if color is not None:
            raise TypeError("Use only one of color and c.")
        color = kwargs.pop("c")
    line_colors, hue_codes, categories, hue_title, mappable = _resolve_color_params(
        color, hue, palette, count=y_array.shape[0], color_type=color_type
    )
    line_styles = _resolve_hue_styles(hue_styles, hue_codes, categories)

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
    for index, (line, line_color) in enumerate(zip(lines, line_colors)):
        line.set_color(line_color)
        if line_styles is not None:
            line.set(**line_styles[index])

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

    # Use the same hue guide behavior for created and supplied axes.
    if hue_codes is not None and legend:
        _add_legend(ax, lines, hue_codes, categories, hue_title)
    elif mappable is not None and legend:
        fig.colorbar(mappable, ax=ax, location="right", label=hue_title or "")

    # Use the final axis limits so existing content and margins are included.
    if symmetric:
        limit = max(abs(value) for value in ax.get_ylim())
        ax.set_ylim(-limit, limit)

    return fig, ax
