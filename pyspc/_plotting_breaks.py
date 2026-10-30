"""Post-process linear Matplotlib axes to omit x-coordinate intervals."""

from numbers import Real

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.scale import FuncScale
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator, NullLocator
from matplotlib.transforms import IdentityTransform, Transform
from numpy.typing import ArrayLike


def _validate_spacing(value: float | None, name: str) -> float | None:
    """Accept a positive, finite data-coordinate spacing or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite number.")
    try:
        spacing = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a positive finite number.") from error
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return spacing


def _set_xtick_spacing(ax: Axes, major: float | None, minor: float | None) -> None:
    """Configure requested intervals without changing an unbroken axis's scale."""
    if major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(major))
    if minor is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(minor))


def _validate_x_breaks(x_breaks: ArrayLike, *, ax: Axes | None = None) -> np.ndarray:
    """Normalize intervals and check an optional target without changing it."""
    intervals = np.asarray(x_breaks)
    if intervals.ndim == 1 and intervals.size == 0:
        intervals = intervals.reshape(0, 2)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("x_breaks must have shape (n_breaks, 2).")
    if intervals.dtype.kind not in "iuf":
        raise TypeError("x_breaks must contain real numeric values.")
    if not np.isfinite(intervals).all():
        raise ValueError("x_breaks must contain finite values.")
    if np.any(intervals[:, 0] >= intervals[:, 1]):
        raise ValueError("Each x_breaks interval must have lower < upper.")
    intervals = intervals[np.argsort(intervals[:, 0])].astype(float)
    if np.any(intervals[1:, 0] <= intervals[:-1, 1]):
        raise ValueError("x_breaks intervals must not overlap or touch.")
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


def break_xaxis(
    ax: Axes,
    x_breaks: ArrayLike,
    *,
    x_breaks_gap: float | None = None,
    xtick_major: float | None = None,
    xtick_minor: float | None = None,
) -> Axes:
    """Omit x-coordinate intervals while keeping the original Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A linear x axis with its data and limits already configured. An axis
        previously processed by this function is also accepted.
    x_breaks : array-like of shape (n_breaks, 2)
        Intervals to omit, in original data coordinates, not ranges to retain.
        Each pair must contain finite values with lower < upper, even for an
        inverted axis. Intervals are sorted and must not overlap or touch.
        Empty input leaves the scale unchanged, but requested tick spacings
        are still applied. Intervals outside the current view are ignored;
        partially visible intervals are clipped to the view. The entire view
        cannot be omitted.
    x_breaks_gap : positive float, optional
        Width of every gap in the same units as the retained x coordinates.
        Defaults to xtick_minor when provided, otherwise half the major-tick
        interval, including when the major interval is chosen automatically.
    xtick_major, xtick_minor : positive float, optional
        Tick spacings in original x-data units, anchored at zero. Ticks inside
        omitted intervals are excluded. Explicit major spacing is honored
        without thinning labels, even if labels become crowded. If None, major
        spacing is automatic and minor ticks are disabled for broken axes.
        With no visible breaks, omitted options leave existing locators intact.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied axis, with retained ranges on one linear scale, configured gaps,
        break marks, and ticks in the original data coordinates. No new axes are
        created. Original line data and legend handles are retained.

    Notes
    -----
    Apply this as a final plotting step. It installs a custom x scale and fixed
    tick locators, and replaces clipping paths on lines, filled polygons, and
    horizontal spines. Line and polygon paths are subdivided
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
    intervals = _validate_x_breaks(x_breaks, ax=ax)
    x_breaks_gap = _validate_spacing(x_breaks_gap, "x_breaks_gap")
    xtick_major = _validate_spacing(xtick_major, "xtick_major")
    xtick_minor = _validate_spacing(xtick_minor, "xtick_minor")
    if not len(intervals):
        _set_xtick_spacing(ax, xtick_major, xtick_minor)
        return ax
    limits = ax.get_xlim()
    lower, upper = sorted(limits)
    if not np.isfinite(upper - lower):
        raise ValueError("The visible x range is too large for x_breaks.")
    intervals = intervals[(intervals[:, 1] > lower) & (intervals[:, 0] < upper)]
    if not len(intervals):
        _set_xtick_spacing(ax, xtick_major, xtick_minor)
        return ax
    intervals = np.clip(intervals, lower, upper)
    widths = intervals[:, 1] - intervals[:, 0]
    retained_width = (upper - lower) - widths.sum()
    if retained_width <= 0:
        raise ValueError("x_breaks cannot omit the entire visible x range.")

    # Determine a global major interval and derive the default gap from ticks.
    major = xtick_major
    if major is None:
        automatic = MaxNLocator(nbins=max(2, ax.xaxis.get_tick_space()), min_n_ticks=2)
        major = float(np.min(np.diff(automatic.tick_values(0, retained_width))))
    gap = x_breaks_gap
    if gap is None:
        gap = xtick_minor if xtick_minor is not None else major / 2
    if gap <= 0:
        raise ValueError("The retained x range is too small for x_breaks.")
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

    # Generate regular data-coordinate ticks only in the retained ranges.
    for interval, setter in (
        (major, ax.xaxis.set_major_locator),
        (xtick_minor, ax.xaxis.set_minor_locator),
    ):
        if interval is None:
            setter(NullLocator())
            continue
        ticks = []
        locator = MultipleLocator(interval)
        for left, right in retained:
            candidates = locator.tick_values(left, right)
            ticks.extend(candidates[(candidates >= left) & (candidates <= right)])
        setter(FixedLocator(np.unique(ticks)))

    # Only automatic major labels may be thinned near compressed gaps.
    if xtick_major is None:
        spacing = (forward(upper) - forward(lower)) / max(2, ax.xaxis.get_tick_space())
        separated = []
        for tick in ax.get_xticks():
            if not separated or forward(tick) - forward(separated[-1]) >= spacing:
                separated.append(tick)
        ax.xaxis.set_major_locator(FixedLocator(separated))

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
