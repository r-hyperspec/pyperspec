import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba, to_rgba_array

from pyspc.plotting import _resolve_color_params, _validate_array, plot_spectra


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


@pytest.mark.parametrize("supplied_axis", [False, True])
def test_plot_uses_one_default_color_for_all_spectra_and_bands(supplied_axis):
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=["red", "blue"])}):
        ax = None
        if supplied_axis:
            _, ax = plt.subplots()
            ax.set_prop_cycle(color=["green", "orange"])
        _, ax = plot_spectra(
            [[1, 2], [3, 4]], ymin=[[0, 1], [2, 3]], ymax=[[2, 3], [4, 5]], ax=ax
        )
        plot_spectra([[5, 6]], ax=ax)

        assert [to_rgba(line.get_color()) for line in ax.lines] == [to_rgba("red")] * 3
        for band in ax.collections:
            np.testing.assert_allclose(band.get_facecolor(), [to_rgba("red", 0.3)])

    # Colors remain matched after the temporary style has been restored.
    assert [to_rgba(line.get_color()) for line in ax.lines] == [to_rgba("red")] * 3


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
        ([[1, 2]], [0, np.nan], ValueError, "x must not contain NaN"),
        ([[1, 2]], [0, np.inf], ValueError, "x must not contain infinity"),
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
    fig, returned_ax = plot_spectra(
        y,
        x=x,
        ymin=ymin,
        ymax=ymax,
        palette=["red", "blue"],
        ax=ax,
        alpha=0.8,
        zorder=0,
    )
    plot_spectra([[1, 2, 3]], color="green", ax=ax)

    assert returned_ax is ax
    assert fig is ax.figure
    assert len(ax.collections) == len(y)
    assert [to_rgba(line.get_color()) for line in ax.lines] == [
        to_rgba("red"),
        to_rgba("blue"),
        to_rgba("green"),
    ]
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
@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("argument", ["color", "c"])
def test_plot_treats_rgb_and_rgba_values_as_one_color(color, count, argument):
    _, ax = plot_spectra(np.ones((count, 2)), **{argument: color})

    for line in ax.lines:
        assert to_rgba(line.get_color()) == to_rgba(color)


@pytest.mark.parametrize(
    "colors",
    [
        ["black"],
        ["red", "blue"],
        [(1, 0, 0), (0, 0, 1)],
        np.array([[1, 0, 0, 0.5], [0, 0, 1, 0.8]]),
    ],
)
@pytest.mark.parametrize("argument", ["color", "c"])
def test_plot_accepts_one_specific_color_per_spectrum_with_matching_bands(
    colors, argument
):
    y = np.ones((len(colors), 2))

    _, ax = plot_spectra(y, ymin=y - 1, ymax=y + 1, **{argument: colors})

    for line, band, expected in zip(ax.lines, ax.collections, to_rgba_array(colors)):
        np.testing.assert_allclose(to_rgba(line.get_color()), expected)
        np.testing.assert_allclose(band.get_facecolor(), [to_rgba(expected, 0.3)])


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

    next_line = ax.plot([0, 1], [7, 8])[0]
    assert next_line.get_color() == "red"


@pytest.mark.parametrize(
    "options, error",
    [
        ({"color": []}, ValueError),
        ({"color": ["red"]}, ValueError),
        ({"color": ["red", "blue", "green"]}, ValueError),
        ({"color": ["red", "invalid"]}, ValueError),
        ({"color": ["red", "blue"], "hue": ["A", "B"]}, ValueError),
        ({"color": ["red", "blue"], "palette": "viridis"}, ValueError),
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
            "ymin must have shape",
        ),
        (
            {"ymin": [[0, 1]], "ymax": [[2]]},
            ValueError,
            "ymax must have shape",
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


def test_plot_maps_hue_categories_to_colors_and_bands_without_aggregation():
    y = np.array([[1, 2], [3, 4], [5, 6]])

    _, ax = plot_spectra(
        y,
        hue=["A", "A", "B"],
        palette={"A": "red", "B": "blue"},
        ymin=y - 1,
        ymax=y + 2,
        linestyle="--",
    )

    assert len(ax.lines) == len(y)
    assert len(ax.collections) == len(y)
    for line, band, spectrum, expected in zip(
        ax.lines, ax.collections, y, ["red", "red", "blue"]
    ):
        np.testing.assert_array_equal(line.get_ydata(), spectrum)
        assert to_rgba(line.get_color()) == to_rgba(expected)
        assert line.get_linestyle() == "--"
        np.testing.assert_allclose(band.get_facecolor(), [to_rgba(expected, 0.3)])


def test_plot_palette_dictionary_keeps_colors_consistent_across_calls():
    palette = {"A": "red", "B": "blue", "unused": "green"}

    _, first = plot_spectra([[1, 2]], hue=["B"], palette=palette)
    _, second = plot_spectra([[3, 4]], hue=["A"], palette=palette)

    assert to_rgba(first.lines[0].get_color()) == to_rgba("blue")
    assert to_rgba(second.lines[0].get_color()) == to_rgba("red")
    assert palette == {"A": "red", "B": "blue", "unused": "green"}


@pytest.mark.parametrize("palette", [["red", "blue"], "viridis"])
def test_plot_uses_pandas_hue_order_with_one_legend_entry_per_category(
    palette,
):
    _, ax = plot_spectra([[1, 2], [3, 4], [5, 6]], hue=["B", "A", "B"], palette=palette)

    expected = (
        to_rgba_array(palette)
        if isinstance(palette, list)
        else plt.get_cmap(palette)([0.0, 1.0])
    )
    for line, color in zip(ax.lines, expected[[1, 0, 1]]):
        np.testing.assert_allclose(to_rgba(line.get_color()), color)
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["A", "B"]
    for handle, color in zip(legend.legend_handles, expected):
        np.testing.assert_allclose(to_rgba(handle.get_color()), color)


@pytest.mark.parametrize("as_series", [False, True])
def test_plot_preserves_declared_categorical_order_without_unused_categories(
    as_series,
):
    hue = pd.Categorical([10, 20, 10], categories=[20, 10, 30], ordered=True)
    if as_series:
        hue = pd.Series(hue, index=[9, 7, 5], name="Treatment")

    _, ax = plot_spectra([[1, 2], [3, 4], [5, 6]], hue=hue, palette=["red", "blue"])

    for line, expected in zip(ax.lines, ["blue", "red", "blue"]):
        assert to_rgba(line.get_color()) == to_rgba(expected)
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["20", "10"]
    assert legend.get_title().get_text() == ("Treatment" if as_series else "")


@pytest.mark.parametrize("hue", [[None, "NA", np.nan, pd.NA], [None] * 4])
def test_plot_keeps_missing_hue_gray_and_separate_from_literal_na(hue):
    _, ax = plot_spectra(np.ones((4, 2)), hue=hue, palette={"NA": "red"})

    for line, value in zip(ax.lines, hue):
        expected = "gray" if pd.isna(value) else "red"
        assert to_rgba(line.get_color()) == to_rgba(expected)
    expected_labels = ["Missing"] if all(pd.isna(hue)) else ["NA", "Missing"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == expected_labels


def test_plot_uses_default_palette_per_hue_category():
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=["red", "blue"])}):
        _, ax = plot_spectra([[1, 2], [3, 4], [5, 6]], hue=["A", "A", "B"])

    assert [to_rgba(line.get_color()) for line in ax.lines] == [
        to_rgba("red"),
        to_rgba("red"),
        to_rgba("blue"),
    ]


def test_plot_can_skip_hue_legend_and_preserve_existing_legend():
    _, ax = plt.subplots()
    ax.plot([0, 1], [1, 2], label="Existing")
    existing = ax.legend()

    plot_spectra([[3, 4]], hue=["A"], legend=False, ax=ax)

    assert ax.get_legend() is existing


@pytest.mark.parametrize(
    "hue",
    [
        [10, 11, 20],
        np.array([10.0, 11.0, 20.0]),
        pd.Series([10, 11, 20], index=[9, 7, 5], name="Dose"),
        pd.Series([10, 11, 20], dtype="Int64", name="Dose"),
    ],
)
@pytest.mark.parametrize("palette", [None, "plasma", plt.get_cmap("plasma")])
def test_plot_maps_numeric_hue_by_value_with_matching_bands(hue, palette):
    y = np.array([[1, 2], [3, 4], [5, 6]])

    fig, ax = plot_spectra(y, hue=hue, palette=palette, ymin=y - 1, ymax=y + 1)

    cmap = plt.get_cmap("viridis" if palette is None else palette)
    expected = cmap([0.0, 0.1, 1.0])
    assert len(ax.lines) == 3
    assert ax.get_legend() is None
    assert fig.axes == [ax]
    for line, band, color in zip(ax.lines, ax.collections, expected):
        np.testing.assert_allclose(to_rgba(line.get_color()), color)
        np.testing.assert_allclose(band.get_facecolor(), [to_rgba(color, 0.3)])


@pytest.mark.parametrize(
    "hue",
    [
        [10, None, 20],
        [10, pd.NA, 20],
        pd.Series([10, pd.NA, 20], dtype="Float64"),
    ],
)
def test_plot_numeric_hue_keeps_missing_values_gray(hue):
    fig, ax = plot_spectra(np.ones((3, 2)), hue=hue)

    expected = [
        plt.get_cmap("viridis")(0.0),
        to_rgba("gray"),
        plt.get_cmap("viridis")(1.0),
    ]
    np.testing.assert_allclose(
        [to_rgba(line.get_color()) for line in ax.lines], expected
    )
    assert fig.axes == [ax]
    assert ax.get_legend() is None


@pytest.mark.parametrize(
    "hue", [np.full(3, np.nan), pd.Series([pd.NA] * 3, dtype="Float64")]
)
def test_plot_all_missing_numeric_hue_draws_gray_without_colorbar(hue):
    fig, ax = plot_spectra(np.ones((3, 2)), hue=hue)

    np.testing.assert_allclose(
        [to_rgba(line.get_color()) for line in ax.lines], [to_rgba("gray")] * 3
    )
    assert fig.axes == [ax]
    assert ax.get_legend() is None


@pytest.mark.parametrize("value", [0, -10, 20])
@pytest.mark.parametrize("count", [1, 2])
def test_plot_constant_numeric_hue_uses_midpoint(value, count):
    fig, ax = plot_spectra(np.ones((count, 2)), hue=[value] * count)

    expected = plt.get_cmap("viridis")(0.5)
    for line in ax.lines:
        np.testing.assert_allclose(to_rgba(line.get_color()), expected)
    assert fig.axes == [ax]


@pytest.mark.parametrize("hue", [[1.0, 1.0 + 5e-16], [0.0, 1e-320]])
def test_plot_nearly_constant_numeric_hue_uses_same_scale_with_legend_disabled(hue):
    fig, ax = plot_spectra(np.ones((2, 2)), hue=hue)
    hidden_fig, hidden_ax = plot_spectra(np.ones((2, 2)), hue=hue, legend=False)
    _, _, _, _, mappable = _resolve_color_params(hue=hue, count=2)

    colors = [to_rgba(line.get_color()) for line in ax.lines]
    np.testing.assert_allclose(colors, mappable.to_rgba(hue))
    np.testing.assert_allclose(
        colors, [to_rgba(line.get_color()) for line in hidden_ax.lines]
    )
    assert fig.axes == [ax]
    assert hidden_fig.axes == [hidden_ax]


def test_plot_can_treat_numeric_hue_as_discrete_categories():
    fig, ax = plot_spectra(
        np.ones((3, 2)),
        hue=[20, 10, 20],
        color_type="discrete",
        palette={10: "red", 20: "blue"},
    )

    np.testing.assert_allclose(
        [to_rgba(line.get_color()) for line in ax.lines],
        to_rgba_array(["blue", "red", "blue"]),
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["10", "20"]
    assert fig.axes == [ax]


@pytest.mark.parametrize("hue", [[True, False, True], ["20", "10", "20"]])
def test_plot_keeps_boolean_and_numeric_string_hue_discrete(hue):
    fig, ax = plot_spectra(np.ones((3, 2)), hue=hue)

    assert fig.axes == [ax]
    assert len(ax.get_legend().get_texts()) == 2
    assert to_rgba(ax.lines[0].get_color()) == to_rgba(ax.lines[2].get_color())


def test_plot_repeated_numeric_values_share_colors_across_a_negative_range():
    _, ax = plot_spectra(np.ones((4, 2)), hue=[-10, 0, 10, -10])

    np.testing.assert_allclose(
        [to_rgba(line.get_color()) for line in ax.lines],
        plt.get_cmap("viridis")([0.0, 0.5, 1.0, 0.0]),
    )


def test_plot_numeric_hue_preserves_supplied_figure_and_other_axes():
    fig, (ax, other) = plt.subplots(1, 2)
    other_line = other.plot([0, 1], [10, 20])[0]
    figures = plt.get_fignums()

    returned_fig, returned_ax = plot_spectra(np.ones((2, 2)), hue=[1, 2], ax=ax)

    assert returned_fig is fig
    assert returned_ax is ax
    assert plt.get_fignums() == figures
    assert fig.axes == [ax, other]
    assert list(other.lines) == [other_line]


@pytest.mark.parametrize("legend", [False, True])
def test_plot_numeric_hue_preserves_existing_axes_and_legend(legend):
    fig, (ax, other) = plt.subplots(1, 2)
    existing_line = ax.plot([0, 1], [1, 2], label="Existing")[0]
    existing_legend = ax.legend()

    returned_fig, returned_ax = plot_spectra(
        np.ones((2, 2)), hue=[1, 2], legend=legend, ax=ax
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert fig.axes == [ax, other]
    assert ax.lines[0] is existing_line
    assert ax.get_legend() is existing_legend
    assert len(other.lines) == 0


@pytest.mark.parametrize(
    "options, error",
    [
        ({"hue": [1, np.inf]}, ValueError),
        ({"hue": [-np.inf, 2]}, ValueError),
        ({"hue": [1j, 2j]}, TypeError),
        ({"hue": [np.finfo(float).max] * 2}, ValueError),
        ({"hue": [-np.finfo(float).max, np.finfo(float).max]}, ValueError),
        ({"hue": [1, 2], "palette": ["red", "blue"]}, ValueError),
        ({"hue": [1, 2], "palette": {1: "red", 2: "blue"}}, ValueError),
        ({"hue": [1, 2], "palette": "invalid"}, ValueError),
        ({"hue": [1, 2], "color_type": "invalid"}, ValueError),
    ],
)
def test_plot_rejects_invalid_numeric_hue_before_drawing(options, error):
    fig, ax = plt.subplots()
    existing = ax.plot([0, 1], [1, 2])[0]
    figures = plt.get_fignums()

    for target in (None, ax):
        with pytest.raises(error):
            plot_spectra(np.ones((2, 2)), ax=target, **options)

    assert plt.get_fignums() == figures
    assert fig.axes == [ax]
    assert list(ax.lines) == [existing]


@pytest.mark.parametrize(
    "options, error",
    [
        ({"hue": "group"}, ValueError),
        ({"hue": [["A"], ["B"]]}, ValueError),
        ({"hue": ["A"]}, ValueError),
        ({"hue": [{"A": 1}, {"B": 2}]}, TypeError),
        ({"hue": ["A", "B"], "palette": {"A": "red"}}, ValueError),
        (
            {"hue": ["A", "B"], "palette": {"A": "red", "B": "invalid"}},
            ValueError,
        ),
        ({"palette": {"A": "red"}}, ValueError),
        ({"hue": ["A", "B"], "color": "red"}, ValueError),
        ({"hue": ["A", "B"], "c": "red"}, ValueError),
    ],
)
def test_plot_rejects_invalid_hue_before_creating_or_modifying_axes(options, error):
    _, ax = plt.subplots()
    existing_line = ax.plot([0, 1], [10, 20])[0]
    figures = plt.get_fignums()

    for target in (None, ax):
        with pytest.raises(error):
            plot_spectra([[1, 2], [3, 4]], ax=target, **options)

    assert plt.get_fignums() == figures
    assert list(ax.lines) == [existing_line]
    assert len(ax.collections) == 0


def test_color_resolver_returns_shared_default_color_without_hue_metadata():
    colors, codes, categories, title, mappable = _resolve_color_params(count=3)

    np.testing.assert_allclose(colors, [to_rgba("C0")] * 3)
    assert codes is None
    assert categories == []
    assert title is None
    assert mappable is None


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, ["C0", "C0"]),
        ({"color": "black"}, ["black", "black"]),
        ({"color": ["red", "blue"]}, ["red", "blue"]),
        ({"palette": ["red", "blue"]}, ["red", "blue"]),
    ],
)
def test_color_resolver_accepts_omitted_hue_for_all_color_sources(options, expected):
    colors, codes, categories, title, mappable = _resolve_color_params(
        hue=None, count=2, **options
    )

    np.testing.assert_allclose(to_rgba_array(colors), to_rgba_array(expected))
    assert codes is None
    assert categories == []
    assert title is None
    assert mappable is None


def test_color_resolver_returns_colors_and_hue_metadata_without_drawing():
    hue = pd.Series(["B", "A", "B", None], name="Group")
    figures = plt.get_fignums()

    colors, codes, categories, title, mappable = _resolve_color_params(
        hue=hue, palette={"A": "red", "B": "blue"}, count=4
    )

    np.testing.assert_allclose(colors, to_rgba_array(["blue", "red", "blue", "gray"]))
    np.testing.assert_array_equal(codes, [1, 0, 1, -1])
    assert categories == ["A", "B"]
    assert title == "Group"
    assert mappable is None
    assert plt.get_fignums() == figures


def test_color_resolver_returns_numeric_scale_without_drawing():
    figures = plt.get_fignums()
    hue = pd.Series([10, 11, 20, np.nan], name="Dose")

    colors, codes, categories, title, mappable = _resolve_color_params(hue=hue, count=4)

    np.testing.assert_allclose(colors[:3], mappable.to_rgba([10, 11, 20]))
    np.testing.assert_allclose(colors[3], to_rgba("gray"))
    assert codes is None
    assert categories == []
    assert title == "Dose"
    assert mappable.norm.vmin == 10
    assert mappable.norm.vmax == 20
    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, [1, np.nan]),
        ({"inf": "ignore"}, [1, np.inf]),
        ({"na": "raise", "inf": "ignore"}, [1, np.inf]),
        ({"ndim": 2, "shape": (1, 2)}, [[1, np.nan]]),
    ],
)
def test_validate_array_accepts_requested_shape_and_missing_value_policy(
    options, expected
):
    result = _validate_array(expected, "values", **options)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "values, options, error",
    [
        ([1, 2], {"ndim": 2}, ValueError),
        ([[1, 2]], {"shape": (2, 2)}, ValueError),
        ([1, np.nan], {"na": "raise"}, ValueError),
        ([1, np.nan], {"na": "raise", "inf": "ignore"}, ValueError),
        ([1, np.inf], {}, ValueError),
        (["a", "b"], {}, TypeError),
        ([1j, 2j], {}, TypeError),
    ],
)
def test_validate_array_rejects_invalid_numeric_arrays(values, options, error):
    with pytest.raises(error, match="values"):
        _validate_array(values, "values", **options)
