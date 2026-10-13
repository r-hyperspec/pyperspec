import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba, to_rgba_array

from pyspc.plotting import plot_spectra


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


@pytest.mark.parametrize("y", [[[1, 2, 3]], [[1, 2, 3], [4, 5, 6]], [[1]]])
def test_plot_draws_each_row_with_default_coordinates(y):
    fig, ax = plot_spectra(y)

    assert isinstance(ax, Axes)
    assert ax.figure is fig
    assert fig.axes == [ax]
    assert len(ax.lines) == len(y)
    assert len(ax.collections) == 0
    for line, spectrum in zip(ax.lines, y):
        np.testing.assert_array_equal(line.get_xdata(), np.arange(len(spectrum)))
        np.testing.assert_array_equal(line.get_ydata(), spectrum)


@pytest.mark.parametrize("x", [[400, 650, 1000], [1000, 650, 400]])
def test_plot_preserves_coordinates_and_missing_intensities(x):
    y = np.array([[1, np.nan, 3], [4, 5, 6]])
    original = y.copy()

    _, ax = plot_spectra(y, x=x)

    for line, spectrum in zip(ax.lines, y):
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), spectrum)
    np.testing.assert_array_equal(y, original)


def test_plot_forwards_line_styles():
    _, ax = plot_spectra(
        [[1, 2], [3, 4]], color="red", linewidth=2.5, linestyle="--", alpha=0.4
    )

    for line in ax.lines:
        assert line.get_color() == "red"
        assert line.get_linewidth() == 2.5
        assert line.get_linestyle() == "--"
        assert line.get_alpha() == 0.4


def test_plot_uses_axis_color_cycle_across_calls():
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=("red", "blue", "green"))

    plot_spectra([[1, 2], [3, 4]], ax=ax)
    plot_spectra([[5, 6]], ax=ax)

    assert [line.get_color() for line in ax.lines] == ["red", "blue", "green"]


def test_plot_draws_into_supplied_axis_without_replacing_existing_lines():
    fig, (target, other) = plt.subplots(1, 2)
    existing_line = target.plot([0, 1], [10, 20])[0]
    target.set_title("Existing plot")
    figures = plt.get_fignums()

    returned_fig, returned_ax = plot_spectra([[1, 2], [3, 4]], ax=target)
    plot_spectra([[5, 6]], ax=target)

    assert returned_fig is fig
    assert returned_ax is target
    assert plt.get_fignums() == figures
    assert len(fig.axes) == 2
    assert len(target.lines) == 4
    assert target.lines[0] is existing_line
    assert target.get_title() == "Existing plot"
    assert len(other.lines) == 0


@pytest.mark.parametrize(
    "y, x, error, message",
    [
        ([1, 2], None, ValueError, "y must be a 2D array"),
        (np.zeros((1, 2, 3)), None, ValueError, "y must be a 2D array"),
        (np.empty((0, 2)), None, ValueError, "at least one spectrum"),
        ([[]], None, ValueError, "at least one coordinate"),
        ([["a", "b"]], None, TypeError, "y must contain real numeric values"),
        ([[1j, 2j]], None, TypeError, "y must contain real numeric values"),
        ([[1, np.inf]], None, ValueError, "y must not contain infinity"),
        ([[1, 2]], [[0, 1]], ValueError, "x must be a 1D array"),
        ([[1, 2]], [0], ValueError, "x must have length 2"),
        ([[1, 2]], ["a", "b"], TypeError, "x must contain real numeric values"),
        ([[1, 2]], [0, 1j], TypeError, "x must contain real numeric values"),
        ([[1, 2]], [0, np.nan], ValueError, "x must contain only finite values"),
        ([[1, 2]], [0, np.inf], ValueError, "x must contain only finite values"),
    ],
)
def test_plot_rejects_invalid_data_before_creating_or_modifying_axes(
    y, x, error, message
):
    fig, ax = plt.subplots()
    existing_line = ax.plot([0, 1], [10, 20])[0]
    figures = plt.get_fignums()

    for target in (None, ax):
        with pytest.raises(error, match=message):
            plot_spectra(y, x=x, ax=target)

    assert plt.get_fignums() == figures
    assert len(fig.axes) == 1
    assert list(ax.lines) == [existing_line]


def test_plot_rejects_invalid_axis():
    with pytest.raises(TypeError, match="ax must be a matplotlib Axes"):
        plot_spectra([[1, 2]], ax="invalid")


@pytest.mark.parametrize("x", [None, [1000, 650, 400]])
def test_plot_draws_one_band_per_spectrum_with_matching_colors(x):
    y = np.array([[2, 4, 3], [5, 6, 4]])
    ymin = y - 1
    ymax = y + 2
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=("red", "blue", "green"))

    fig, returned_ax = plot_spectra(
        y, x=x, ymin=ymin, ymax=ymax, ax=ax, alpha=0.8, zorder=0
    )
    plot_spectra([[1, 2, 3]], ax=ax)

    assert returned_ax is ax
    assert fig is ax.figure
    assert len(ax.collections) == len(y)
    assert [line.get_color() for line in ax.lines] == ["red", "blue", "green"]
    assert ax.get_ylim()[0] <= ymin.min()
    assert ax.get_ylim()[1] >= ymax.max()
    coordinates = np.arange(y.shape[1]) if x is None else x
    for index, band in enumerate(ax.collections):
        np.testing.assert_allclose(
            band.get_facecolor(), [to_rgba(ax.lines[index].get_color(), alpha=0.3)]
        )
        assert band.get_zorder() < ax.lines[index].get_zorder()
        vertices = band.get_paths()[0].vertices
        for coordinate, lower, upper in zip(coordinates, ymin[index], ymax[index]):
            values = vertices[vertices[:, 0] == coordinate, 1]
            assert values.min() == lower
            assert values.max() == upper


@pytest.mark.parametrize("color_key", ["color", "facecolor", "facecolors", "fc"])
def test_plot_styles_bands_independently_without_changing_fill_options(color_key):
    options = {color_key: "blue", "alpha": 0.2, "zorder": -2}

    _, ax = plot_spectra(
        [[2, 3]],
        ymin=[[1, 2]],
        ymax=[[3, 4]],
        color="red",
        alpha=0.8,
        linewidth=4,
        fill_kws=options,
    )

    assert ax.lines[0].get_color() == "red"
    assert ax.lines[0].get_alpha() == 0.8
    assert ax.lines[0].get_linewidth() == 4
    np.testing.assert_allclose(
        ax.collections[0].get_facecolor(), [to_rgba("blue", 0.2)]
    )
    assert ax.collections[0].get_zorder() == -2
    assert options == {color_key: "blue", "alpha": 0.2, "zorder": -2}


@pytest.mark.parametrize("color", ["#2364AA", (0.2, 0.4, 0.6, 0.8)])
def test_plot_inherits_explicit_line_color_for_every_band(color):
    options = {"alpha": 0.2}

    _, ax = plot_spectra(
        [[2, 3], [4, 5]],
        ymin=[[1, 2], [3, 4]],
        ymax=[[3, 4], [5, 6]],
        color=color,
        fill_kws=options,
    )

    for line, band in zip(ax.lines, ax.collections):
        assert to_rgba(line.get_color()) == to_rgba(color)
        np.testing.assert_allclose(band.get_facecolor(), [to_rgba(color, alpha=0.2)])
    assert options == {"alpha": 0.2}


@pytest.mark.parametrize(
    "palette",
    [
        ["tab:blue", "tab:orange"],
        [(1, 0, 0), (0, 0, 1)],
        plt.get_cmap("viridis")(np.linspace(0, 1, 2)),
    ],
)
def test_plot_accepts_palette_colors_with_matching_bands(palette):
    y = np.array([[2, 3], [4, 5]])

    _, ax = plot_spectra(
        y,
        ymin=y - 1,
        ymax=y + 2,
        label=["first", "second"],
        linewidth=2,
        palette=palette,
    )

    assert len(ax.lines) == len(y)
    for line, band, expected in zip(ax.lines, ax.collections, to_rgba_array(palette)):
        np.testing.assert_allclose(to_rgba(line.get_color()), expected)
        np.testing.assert_allclose(band.get_facecolor(), [to_rgba(expected, 0.3)])
        assert line.get_linewidth() == 2
    assert [line.get_label() for line in ax.lines] == ["first", "second"]


@pytest.mark.parametrize("color", [(0.2, 0.4, 0.6), [0.2, 0.4, 0.6, 0.8]])
@pytest.mark.parametrize("argument", ["color", "c"])
def test_plot_treats_rgb_and_rgba_values_as_one_color(color, argument):
    _, ax = plot_spectra([[1, 2], [3, 4], [5, 6]], **{argument: color})

    for line in ax.lines:
        assert to_rgba(line.get_color()) == to_rgba(color)


@pytest.mark.parametrize("palette", ["viridis", plt.get_cmap("viridis")])
def test_plot_samples_colormaps_in_spectrum_order(palette):
    y = np.array([[2, 3], [4, 5]])

    _, ax = plot_spectra(y, ymin=y - 1, ymax=y + 2, palette=palette)

    expected_colors = plt.get_cmap("viridis")([0.0, 1.0])
    for line, band, expected in zip(ax.lines, ax.collections, expected_colors):
        np.testing.assert_allclose(to_rgba(line.get_color()), expected)
        np.testing.assert_allclose(band.get_facecolor(), [to_rgba(expected, 0.3)])


@pytest.mark.parametrize(
    "palette, expected",
    [
        (["black"], ["black", "black", "black"]),
        (["red", "blue"], ["red", "blue", "red"]),
        (["red", "blue", "green"], ["red", "blue"]),
    ],
)
def test_plot_cycles_or_truncates_explicit_palettes(palette, expected):
    _, ax = plot_spectra(np.ones((len(expected), 2)), palette=palette)

    np.testing.assert_allclose(
        [to_rgba(line.get_color()) for line in ax.lines], to_rgba_array(expected)
    )


def test_plot_samples_colormap_for_single_spectrum():
    _, ax = plot_spectra([[1, 2]], palette="viridis")

    np.testing.assert_allclose(ax.lines[0].get_color(), plt.get_cmap("viridis")(0.0))


def test_plot_explicit_palette_does_not_advance_axis_color_cycle():
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=["red", "green"])

    plot_spectra([[1, 2], [3, 4]], palette=["blue", "orange"], ax=ax)
    plot_spectra([[5, 6]], ax=ax)

    assert ax.lines[-1].get_color() == "red"


@pytest.mark.parametrize(
    "options, error",
    [
        ({"color": []}, ValueError),
        ({"color": ["red"]}, ValueError),
        ({"color": ["red", "blue", "green"]}, ValueError),
        ({"color": ["red", "invalid"]}, ValueError),
        ({"color": "viridis"}, ValueError),
        ({"color": "red", "c": "blue"}, TypeError),
        ({"palette": []}, ValueError),
        ({"palette": ["red", "invalid"]}, ValueError),
        ({"palette": "invalid"}, ValueError),
        ({"palette": "black"}, ValueError),
        ({"color": "red", "palette": "viridis"}, ValueError),
        ({"c": "red", "palette": "viridis"}, ValueError),
    ],
)
def test_plot_rejects_invalid_colors_before_creating_or_modifying_axes(options, error):
    _, ax = plt.subplots()
    existing_line = ax.plot([0, 1], [10, 20])[0]
    figures = plt.get_fignums()

    for target in (None, ax):
        with pytest.raises(error):
            plot_spectra([[1, 2], [3, 4]], ax=target, **options)

    assert plt.get_fignums() == figures
    assert list(ax.lines) == [existing_line]


@pytest.mark.parametrize("fill_kws", [None, {"color": "blue"}])
def test_plot_draws_filled_bands_without_bound_lines_or_outlines(fill_kws):
    y = np.array([[2, 3], [4, 5]])
    with plt.rc_context({"patch.force_edgecolor": True, "patch.linewidth": 3}):
        _, ax = plot_spectra(y, ymin=y - 1, ymax=y + 2, fill_kws=fill_kws)

    assert len(ax.lines) == len(y)
    assert len(ax.collections) == len(y)
    for line, spectrum, band in zip(ax.lines, y, ax.collections):
        np.testing.assert_array_equal(line.get_ydata(), spectrum)
        np.testing.assert_array_equal(band.get_linewidths(), [0])


@pytest.mark.parametrize("width_key", ["linewidth", "linewidths", "lw"])
def test_plot_allows_explicit_band_outline_styles(width_key):
    _, ax = plot_spectra(
        [[2, 3]],
        ymin=[[1, 2]],
        ymax=[[3, 4]],
        fill_kws={"edgecolor": "red", width_key: 2},
    )

    np.testing.assert_array_equal(ax.collections[0].get_linewidths(), [2])
    np.testing.assert_allclose(ax.collections[0].get_edgecolor(), [to_rgba("red", 0.3)])


@pytest.mark.parametrize("missing_bound", ["ymin", "ymax"])
def test_plot_leaves_gaps_where_band_bounds_are_missing(missing_bound):
    bounds = {"ymin": np.zeros((1, 5)), "ymax": np.full((1, 5), 2.0)}
    bounds[missing_bound][0, 2] = np.nan

    _, ax = plot_spectra([[1, 1, 1, 1, 1]], **bounds)

    paths = ax.collections[0].get_paths()
    assert len(paths) == 2
    assert paths[0].vertices[:, 0].max() == 1
    assert paths[1].vertices[:, 0].min() == 3


@pytest.mark.parametrize(
    "ymin, ymax",
    [([[1, 1]], [[1, 1]]), ([[0, 0]], [[1, 1]]), ([[np.nan, np.nan]], [[1, 1]])],
)
def test_plot_accepts_equal_missing_or_independent_bounds(ymin, ymax):
    _, ax = plot_spectra([[2, 3]], ymin=ymin, ymax=ymax)

    assert len(ax.collections) == 1
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2, 3])


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"ymin": [[0, 1]]}, ValueError, "ymin and ymax must be supplied together"),
        ({"ymax": [[2, 3]]}, ValueError, "ymin and ymax must be supplied together"),
        (
            {"ymin": [0, 1], "ymax": [[2, 3]]},
            ValueError,
            "ymin must have the same shape as y",
        ),
        (
            {"ymin": [[0, 1]], "ymax": [[2]]},
            ValueError,
            "ymax must have the same shape as y",
        ),
        (
            {"ymin": [["a", "b"]], "ymax": [[2, 3]]},
            TypeError,
            "ymin must contain real numeric values",
        ),
        (
            {"ymin": [[0, 1]], "ymax": [[2j, 3j]]},
            TypeError,
            "ymax must contain real numeric values",
        ),
        (
            {"ymin": [[0, -np.inf]], "ymax": [[2, 3]]},
            ValueError,
            "ymin must not contain infinity",
        ),
        (
            {"ymin": [[0, 1]], "ymax": [[2, np.inf]]},
            ValueError,
            "ymax must not contain infinity",
        ),
        (
            {"ymin": [[0, 4]], "ymax": [[2, 3]]},
            ValueError,
            "ymin must not exceed ymax",
        ),
        ({"fill_kws": {"alpha": 0.2}}, ValueError, "fill_kws requires band bounds"),
    ],
)
def test_plot_rejects_invalid_bands_before_creating_or_modifying_axes(
    options, error, message
):
    _, ax = plt.subplots()
    existing_line = ax.plot([0, 1], [10, 20])[0]
    figures = plt.get_fignums()

    for target in (None, ax):
        with pytest.raises(error, match=message):
            plot_spectra([[1, 2]], ax=target, **options)

    assert plt.get_fignums() == figures
    assert list(ax.lines) == [existing_line]
    assert len(ax.collections) == 0
