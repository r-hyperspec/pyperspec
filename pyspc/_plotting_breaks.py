"""Post-process linear Matplotlib axes to omit x-coordinate intervals."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.scale import FuncScale
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator
from matplotlib.transforms import IdentityTransform, Transform
from numpy.typing import ArrayLike


def _validate_xbreaks(xbreaks: ArrayLike, *, ax: Axes | None = None) -> np.ndarray:
    """Normalize intervals and check an optional target without changing it."""
    intervals = np.asarray(xbreaks)
    if intervals.ndim == 1 and intervals.size == 0:
        intervals = intervals.reshape(0, 2)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("xbreaks must have shape (n_breaks, 2).")
    if intervals.dtype.kind not in "iuf":
        raise TypeError("xbreaks must contain real numeric values.")
    if not np.isfinite(intervals).all():
        raise ValueError("xbreaks must contain finite values.")
    if np.any(intervals[:, 0] >= intervals[:, 1]):
        raise ValueError("Each xbreaks interval must have lower < upper.")
    intervals = intervals[np.argsort(intervals[:, 0])].astype(float)
    if np.any(intervals[1:, 0] <= intervals[:-1, 1]):
        raise ValueError("xbreaks intervals must not overlap or touch.")
    if len(intervals) and ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError("ax must be a matplotlib Axes.")
        if ax.get_xscale() not in ("linear", _XBreakScale.name):
            raise ValueError("break_xaxis requires a linear x axis.")
    return intervals


class _XBreakScale(FuncScale):
    """Identify our function scale when decorating shared axes or reapplying."""

    name = "pyspc_xbreak"


class _XBreakPathTransform(Transform):
    """Insert boundary vertices before the nonlinear x scale transforms paths."""

    input_dims = output_dims = 2
    is_separable = True
    has_inverse = True

    def __init__(self, boundaries: np.ndarray) -> None:
        super().__init__()
        self.boundaries = boundaries

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:
        return values

    def inverted(self) -> IdentityTransform:
        return IdentityTransform()

    def transform_path_non_affine(self, path: Path) -> Path:
        vertices, codes = [], []
        previous = start = None
        for segment, code in path.iter_segments(
            curves=False, simplify=False, remove_nans=False
        ):
            endpoint = start if code == Path.CLOSEPOLY else segment[-2:]
            if code != Path.MOVETO and previous is not None:
                if np.isfinite(previous).all() and np.isfinite(endpoint).all():
                    dx = endpoint[0] - previous[0]
                    if dx != 0:
                        fractions = (self.boundaries - previous[0]) / dx
                        for fraction in np.sort(
                            fractions[(fractions > 0) & (fractions < 1)]
                        ):
                            vertices.append(previous + fraction * (endpoint - previous))
                            codes.append(Path.LINETO)
            vertices.append(endpoint)
            codes.append(code)
            previous = np.asarray(endpoint)
            if code == Path.MOVETO:
                start = previous
        return Path(vertices, codes) if vertices else path


def break_xaxis(ax: Axes, xbreaks: ArrayLike) -> Axes:
    """Omit x-coordinate intervals while keeping the original Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A linear x axis with its data and limits already configured. An axis
        previously processed by this function is also accepted.
    xbreaks : array-like of shape (n_breaks, 2)
        Intervals to omit, in original data coordinates, not ranges to retain.
        Each pair must contain finite values with lower < upper, even for an
        inverted axis. Intervals are sorted and must not overlap or touch.
        Empty input is a no-op. Intervals outside the current view are ignored;
        partially visible intervals are clipped to the view. The entire view
        cannot be omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied axis, with retained ranges on one linear scale, small gaps,
        break marks, and ticks in the original data coordinates. No new axes are
        created. Original line data and legend handles are retained.

    Notes
    -----
    Apply this as a final plotting step. It installs a custom x scale and fixed
    tick locators, disables minor ticks, and replaces clipping paths on lines,
    filled polygons, and horizontal spines. Line and polygon paths are subdivided
    at break boundaries so sparse data interpolate correctly beside each gap.
    Scatter, images, annotations, and custom artist clipping are not supported.

    Normal axis operations have limitations after breaking: adding artists,
    changing limits, autoscaling, or interactive pan/zoom can leave clipping and
    ticks out of date. Reapply this function after such changes. Repeated calls
    replace the marks and clipping rather than adding duplicate decorations.
    Setting another x scale replaces the break transform and is not an undo
    operation. An empty interval list does not undo existing breaks either.

    X scales are shared by Matplotlib siblings. Apply this function to every
    visible axis in a shared group to install each panel's clipping and marks.
    Changing from a logarithmic or other nonlinear scale is rejected.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pyspc.plotting import break_xaxis
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([0, 10], [0, 10])
    >>> _ = ax.set_xlim(0, 10)
    >>> _ = break_xaxis(ax, [(4, 8)])
    >>> plt.close(fig)
    """
    # Validate before changing artists or shared axis state.
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes.")
    intervals = _validate_xbreaks(xbreaks, ax=ax)
    if not len(intervals):
        return ax
    limits = ax.get_xlim()
    lower, upper = sorted(limits)
    if not np.isfinite(upper - lower):
        raise ValueError("The visible x range is too large for xbreaks.")
    intervals = intervals[(intervals[:, 1] > lower) & (intervals[:, 0] < upper)]
    if not len(intervals):
        return ax
    intervals = np.clip(intervals, lower, upper)
    widths = intervals[:, 1] - intervals[:, 0]
    retained_width = (upper - lower) - widths.sum()
    if retained_width <= 0:
        raise ValueError("xbreaks cannot omit the entire visible x range.")

    # Use invertible compression: every omitted interval becomes a small gap.
    gap = retained_width * 0.02
    if gap <= 0:
        raise ValueError("The retained x range is too small for xbreaks.")
    reductions = widths - gap
    starts = intervals[:, 0] - lower
    compressed_starts = starts - (np.cumsum(reductions) - reductions)

    def forward(values: ArrayLike) -> ArrayLike:
        coordinates = np.asarray(values, dtype=float) - lower
        result = coordinates.copy()
        for start, width, reduction in zip(starts, widths, reductions):
            result -= np.clip((coordinates - start) / width, 0, 1) * reduction
        return result

    def inverse(values: ArrayLike) -> ArrayLike:
        coordinates = np.asarray(values, dtype=float)
        result = coordinates + lower
        for start, reduction in zip(compressed_starts, reductions):
            result = result + np.clip((coordinates - start) / gap, 0, 1) * reduction
        return result

    # Keep the axis view and its autoscale setting while replacing its x scale.
    autoscale = ax.get_autoscalex_on()
    ax.set_xscale(_XBreakScale(ax.xaxis, functions=(forward, inverse)))
    ax.set_xlim(limits, auto=autoscale)
    edges = np.concatenate(([lower], intervals.ravel(), [upper])).reshape(-1, 2)
    retained = edges[edges[:, 1] > edges[:, 0]]

    # Choose readable data-coordinate ticks independently within retained ranges.
    ticks = []
    for left, right in retained:
        bins = max(1, round(6 * (right - left) / retained_width))
        candidates = MaxNLocator(nbins=bins, min_n_ticks=2).tick_values(left, right)
        ticks.extend(candidates[(candidates >= left) & (candidates <= right)])
    # Ticks on opposite sides of a compressed gap must not crowd each other.
    spacing = (forward(upper) - forward(lower)) / max(2, ax.xaxis.get_tick_space())
    separated = []
    for tick in np.unique(ticks):
        if not separated or forward(tick) - forward(separated[-1]) >= spacing:
            separated.append(tick)
    ax.xaxis.set_major_locator(FixedLocator(separated))
    ax.xaxis.set_minor_locator(NullLocator())

    # Clip visible segments, interpolating original line/band paths at boundaries.
    clip_path = Path.make_compound_path_from_polys(
        np.array(
            [[[left, 0], [left, 1], [right, 1], [right, 0]] for left, right in retained]
        )
    )
    clip_transform = ax.get_xaxis_transform()
    for line in list(ax.lines):
        if line.get_gid() == "pyspc-xbreak-marks":
            line.remove()
    artists = [
        *ax.lines,
        *(artist for artist in ax.collections if isinstance(artist, PolyCollection)),
    ]
    for artist in artists:
        if not hasattr(artist, "_pyspc_xbreak_base_transform"):
            artist._pyspc_xbreak_base_transform = artist.get_transform()
        artist.set_transform(
            _XBreakPathTransform(intervals.ravel())
            + artist._pyspc_xbreak_base_transform
        )
        artist.set_clip_path(clip_path, clip_transform)
    for name in ("bottom", "top"):
        ax.spines[name].set_clip_path(clip_path, clip_transform)

    # Marker sizes are in points so resizing a figure does not stretch the slashes.
    for name, height in (("bottom", 0), ("top", 1)):
        if ax.spines[name].get_visible():
            marks = ax.plot(
                intervals.ravel(),
                np.full(intervals.size, height),
                transform=clip_transform,
                linestyle="none",
                marker=[(-1, -0.5), (1, 0.5)],
                markersize=6,
                color=ax.spines[name].get_edgecolor(),
                clip_on=False,
                scalex=False,
                scaley=False,
                label="_nolegend_",
            )[0]
            marks.set_gid("pyspc-xbreak-marks")
    return ax
