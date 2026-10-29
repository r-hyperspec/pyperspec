"""Public behavior of spectral facet grids."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba, to_rgba_array

from pyspc.plotting import plot_spectra


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


def test_facets_split_spectra_by_position_and_keep_empty_combinations():
    y = np.arange(12).reshape(4, 3)
    row = pd.Series(["B", "A", "B", "B"], index=[4, 3, 2, 1], name="Treatment")
    col = pd.Series([2, 1, 1, 2], index=[1, 2, 3, 4], name="Time")

    fig, axes = plot_spectra(y, row=row, col=col)

    assert axes.shape == (2, 2)
    assert fig.axes == list(axes.flat)
    assert [[ax.get_title() for ax in group] for group in axes] == [
        ["Treatment = A | Time = 1", "Treatment = A | Time = 2"],
        ["Treatment = B | Time = 1", "Treatment = B | Time = 2"],
    ]
    for ax, indices in zip(axes.flat, [[1], [], [2], [0, 3]]):
        assert ax.get_visible()
        assert len(ax.lines) == len(indices)
        for line, index in zip(ax.lines, indices):
            np.testing.assert_array_equal(line.get_ydata(), y[index])


@pytest.mark.parametrize("argument, shape", [("row", (2, 1)), ("col", (1, 2))])
@pytest.mark.parametrize("as_series", [False, True])
def test_facets_preserve_categorical_order_and_omit_unused_levels(
    argument, shape, as_series
):
    membership = pd.Categorical(["A", "B", "A"], categories=["B", "unused", "A"])
    if as_series:
        membership = pd.Series(membership, name="Group", index=[8, 4, 2])

    _, axes = plot_spectra([[1, 2], [3, 4], [5, 6]], **{argument: membership})

    assert axes.shape == shape
    prefix = "Group = " if as_series else ""
    assert [ax.get_title() for ax in axes.flat] == [prefix + "B", prefix + "A"]
    assert [len(ax.lines) for ax in axes.flat] == [1, 2]


@pytest.mark.parametrize("argument", ["row", "col"])
@pytest.mark.parametrize(
    "membership, titles, counts",
    [
        ([None, "A", np.nan, pd.NA], ["A", "Missing"], [1, 3]),
        ([None, "Missing", "NA", None], ["Missing", "NA", "Missing"], [1, 1, 2]),
        ([None, np.nan, pd.NA, None], ["Missing"], [4]),
        (
            pd.Series([1, pd.NA, 2, pd.NA], dtype="Int64"),
            ["1", "2", "Missing"],
            [1, 1, 2],
        ),
    ],
)
def test_facets_keep_missing_membership_separate_from_literal_labels(
    argument, membership, titles, counts
):
    _, axes = plot_spectra(np.arange(8).reshape(4, 2), **{argument: membership})

    assert axes.ndim == 2
    assert [ax.get_title() for ax in axes.flat] == titles
    assert [len(ax.lines) for ax in axes.flat] == counts


def test_facets_retain_spectra_missing_both_memberships():
    _, axes = plot_spectra([[1, 2], [3, 4]], row=["A", None], col=[None, "B"])

    assert axes.shape == (2, 2)
    assert [ax.get_title() for ax in axes.flat] == [
        "A | B",
        "A | Missing",
        "Missing | B",
        "Missing | Missing",
    ]
    assert [len(ax.lines) for ax in axes.flat] == [0, 1, 1, 0]


@pytest.mark.parametrize("argument", ["row", "col"])
def test_one_level_facets_return_a_two_dimensional_axes_array(argument):
    fig, axes = plot_spectra([[1, 2]], **{argument: ["A"]})

    assert axes.shape == (1, 1)
    assert fig.axes == [axes[0, 0]]
    assert axes[0, 0].get_title() == "A"


@pytest.mark.parametrize("col_wrap", [None, 1])
def test_facets_slice_bands_and_forward_styles_without_mutating_inputs(col_wrap):
    y = np.array([[2, 4, np.nan], [5, 6, 4], [8, 9, 7]])
    lower, upper = y - 2, y + 3
    original = y.copy()
    fill_kws = {"alpha": 0.2, "edgecolor": "black"}
    x = [1000, 700, 400]

    _, axes = plot_spectra(
        y,
        x=x,
        col=["B", "A", "B"],
        col_wrap=col_wrap,
        ymin=lower,
        ymax=upper,
        color=["red", "blue", "green"],
        linestyle="--",
        linewidth=2,
        fill_kws=fill_kws,
        sharey=False,
    )

    for ax, indices in zip(axes.flat, [[1], [0, 2]]):
        assert len(ax.collections) == len(indices)
        for line, band, index in zip(ax.lines, ax.collections, indices):
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y[index])
            assert line.get_linestyle() == "--"
            assert line.get_linewidth() == 2
            np.testing.assert_allclose(
                band.get_facecolor(), [to_rgba(line.get_color(), 0.2)]
            )
            vertices = np.concatenate([path.vertices for path in band.get_paths()])
            for coordinate, low, high in zip(x, lower[index], upper[index]):
                if np.isnan(low):
                    continue
                values = vertices[vertices[:, 0] == coordinate, 1]
                assert values.min() == low
                assert values.max() == high
    np.testing.assert_array_equal(y, original)
    assert fill_kws == {"alpha": 0.2, "edgecolor": "black"}


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, to_rgba_array(["C0"] * 4)),
        ({"color": "black"}, to_rgba_array(["black"] * 4)),
        ({"c": (0.2, 0.3, 0.4)}, to_rgba_array([(0.2, 0.3, 0.4)] * 4)),
        (
            {"color": ["red", "blue", "green", "black"]},
            to_rgba_array(["red", "blue", "green", "black"]),
        ),
        (
            {"palette": ["red", "blue", "green"]},
            to_rgba_array(["red", "blue", "green", "red"]),
        ),
        ({"palette": "viridis"}, plt.get_cmap("viridis")(np.linspace(0, 1, 4))),
    ],
)
@pytest.mark.parametrize("col_wrap", [None, 1])
def test_facets_resolve_literal_colors_and_palettes_in_global_spectrum_order(
    options, expected, col_wrap
):
    fig, axes = plot_spectra(
        np.ones((4, 2)), col=["B", "A", "B", "A"], col_wrap=col_wrap, **options
    )

    for ax, indices in zip(axes.flat, [[1, 3], [0, 2]]):
        np.testing.assert_allclose(
            [to_rgba(line.get_color()) for line in ax.lines], expected[indices]
        )
    assert not fig.legends


@pytest.mark.parametrize("legend", [False, True])
@pytest.mark.parametrize("col_wrap", [None, 2])
def test_facets_use_global_hue_colors_and_one_outside_legend(legend, col_wrap):
    y = np.arange(10).reshape(5, 2)
    hue = pd.Series(
        pd.Categorical(
            ["B", "A", "B", None, "C"], categories=["C", "B", "A", "unused"]
        ),
        name="Treatment",
    )
    palette = ["red", "blue", "green"]

    fig, axes = plot_spectra(
        y,
        col=["right", "left", "left", None, "right"],
        col_wrap=col_wrap,
        hue=hue,
        palette=palette,
        legend=legend,
        ymin=y - 1,
        ymax=y + 1,
        linestyle="--",
        linewidth=3,
    )

    expected = to_rgba_array(["blue", "green", "blue", "gray", "red"])
    for ax, indices in zip(axes.flat, [[1, 2], [0, 4], [3]]):
        assert ax.get_legend() is None
        for line, band, index in zip(ax.lines, ax.collections, indices):
            np.testing.assert_allclose(to_rgba(line.get_color()), expected[index])
            np.testing.assert_allclose(
                band.get_facecolor(), [to_rgba(expected[index], 0.3)]
            )
    assert len(fig.legends) == int(legend)
    if legend:
        shared_legend = fig.legends[0]
        assert shared_legend.get_title().get_text() == "Treatment"
        assert [text.get_text() for text in shared_legend.get_texts()] == [
            "C",
            "B",
            "A",
            "Missing",
        ]
        np.testing.assert_allclose(
            [to_rgba(handle.get_color()) for handle in shared_legend.legend_handles],
            to_rgba_array(["red", "blue", "green", "gray"]),
        )
        assert all(
            handle.get_linestyle() == "--" for handle in shared_legend.legend_handles
        )
        assert all(
            handle.get_linewidth() == 3 for handle in shared_legend.legend_handles
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = shared_legend.get_window_extent(renderer)
        figure_box = fig.get_window_extent(renderer)
        assert legend_box.x0 >= max(
            ax.get_window_extent(renderer).x1 for ax in axes.flat
        )
        assert figure_box.contains(legend_box.x0, legend_box.y0)
        assert figure_box.contains(legend_box.x1, legend_box.y1)


@pytest.mark.parametrize("hue", [[None, None], [10, 20], [10, None]])
@pytest.mark.parametrize("col_wrap", [None, 1])
def test_facets_use_global_continuous_hue_scale_or_all_missing_legend(hue, col_wrap):
    fig, axes = plot_spectra(
        [[1, 2], [3, 4]], col=["A", "B"], col_wrap=col_wrap, hue=hue
    )
    _, reference = plot_spectra([[1, 2], [3, 4]], hue=hue)

    np.testing.assert_allclose(
        [to_rgba(ax.lines[0].get_color()) for ax in axes.flat],
        [to_rgba(line.get_color()) for line in reference.lines],
    )
    has_colorbar = hue != [None, None]
    assert fig.axes[: axes.size] == list(axes.flat)
    assert len(fig.axes) == axes.size + int(has_colorbar)
    assert len(fig.legends) == int(hue == [None, None])


@pytest.mark.parametrize("sharex", [True, False, "row", "col"])
@pytest.mark.parametrize("sharey", [True, False, "row", "col"])
def test_facets_share_axes_in_the_requested_groups(sharex, sharey):
    _, axes = plot_spectra(
        np.arange(8).reshape(4, 2),
        row=["A", "A", "B", "B"],
        col=["X", "Y", "X", "Y"],
        sharex=sharex,
        sharey=sharey,
    )

    for coordinate, sharing in [("x", sharex), ("y", sharey)]:
        for (r1, c1), first in np.ndenumerate(axes):
            for (r2, c2), second in np.ndenumerate(axes):
                if first is second:
                    continue
                expected = (
                    sharing is True
                    or (sharing == "row" and r1 == r2)
                    or (sharing == "col" and c1 == c2)
                )
                group = getattr(first, f"get_shared_{coordinate}_axes")()
                assert group.joined(first, second) == expected


def test_facets_share_axes_by_default_and_keep_all_data_with_symmetric_limits():
    y = np.array([[1, 2], [100, 200]])
    _, reference = plot_spectra(y, col=["A", "B"], ymin=y - 5, ymax=y + 50)
    _, axes = plot_spectra(y, col=["A", "B"], ymin=y - 5, ymax=y + 50, symmetric=True)

    assert axes[0, 0].get_shared_x_axes().joined(*axes.flat)
    assert axes[0, 0].get_shared_y_axes().joined(*axes.flat)
    limit = max(abs(value) for value in reference[0, 0].get_ylim())
    assert limit >= 250
    for ax in axes.flat:
        np.testing.assert_allclose(ax.get_ylim(), [-limit, limit])


@pytest.mark.parametrize("sharey", [False, "row", "col"])
def test_facets_apply_symmetric_limits_after_each_shared_group_has_been_drawn(sharey):
    y = np.array([[1, 2], [10, 20], [-300, -200], [1000, 2000]])
    options = {
        "row": ["A", "A", "B", "B"],
        "col": ["X", "Y", "X", "Y"],
        "sharey": sharey,
    }
    _, reference = plot_spectra(y, **options)
    _, axes = plot_spectra(y, symmetric=True, **options)

    for ax, other in zip(axes.flat, reference.flat):
        limit = max(abs(value) for value in other.get_ylim())
        np.testing.assert_allclose(ax.get_ylim(), [-limit, limit])


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"row": "group"}, ValueError, "row"),
        ({"col": [["A"], ["B"]]}, ValueError, "col"),
        ({"row": ["A"]}, ValueError, "row"),
        ({"col": [{"A": 1}, {"B": 2}]}, TypeError, "unhashable"),
        ({"sharex": "invalid"}, ValueError, "sharex"),
        ({"sharey": "invalid"}, ValueError, "sharey"),
        ({"x": [1]}, ValueError, "x"),
        ({"ymin": [[0, 0], [0, 0]]}, ValueError, "together"),
        ({"ymin": [[0, 0], [0, 0]], "ymax": [[1, 1]]}, ValueError, "ymax"),
        ({"ymin": [[0, 0], [5, 5]], "ymax": [[1, 1], [2, 2]]}, ValueError, "exceed"),
        ({"fill_kws": {"alpha": 0.2}}, ValueError, "bounds"),
        ({"hue": ["A"]}, ValueError, "hue"),
        ({"hue": [1, np.inf]}, ValueError, "infinity"),
        ({"color": "red", "c": "blue"}, TypeError, "color and c"),
        ({"hue": ["A", "B"], "color": "red"}, ValueError, "color and hue"),
        ({"hue": ["A", "B"], "palette": {"A": "red"}}, ValueError, "missing hue"),
    ],
)
def test_invalid_facet_inputs_do_not_create_figures(options, error, message):
    figures = plt.get_fignums()
    arguments = {"col": ["A", "B"], **options}

    with pytest.raises(error, match=message):
        plot_spectra([[1, 2], [3, 4]], **arguments)

    assert plt.get_fignums() == figures


def test_facets_reject_supplied_axis_without_modifying_it():
    fig, ax = plt.subplots()
    existing = ax.plot([0, 1], [1, 2])[0]
    figures = plt.get_fignums()

    with pytest.raises(ValueError, match="ax.*facet"):
        plot_spectra([[1, 2], [3, 4]], row=["A", "B"], ax=ax)

    assert plt.get_fignums() == figures
    assert fig.axes == [ax]
    assert list(ax.lines) == [existing]


@pytest.mark.parametrize("col_wrap", [1, 2, 3, 4, 5, 7, np.int64(2)])
def test_wrapped_facets_fill_in_order_and_hide_spare_axes(col_wrap):
    y = np.arange(18).reshape(6, 3)
    col = pd.Series(
        ["E", "B", "C", "A", "D", "B"], index=[6, 5, 4, 3, 2, 1], name="Group"
    )

    fig, axes = plot_spectra(y, col=col, col_wrap=col_wrap)

    assert axes.shape == ((5 + col_wrap - 1) // col_wrap, col_wrap)
    assert fig.axes == list(axes.flat)
    for ax, title, indices in zip(axes.flat, "ABCDE", [[3], [1, 5], [2], [4], [0]]):
        assert ax.get_visible()
        assert ax.get_title() == f"Group = {title}"
        assert len(ax.lines) == len(indices)
        for line, index in zip(ax.lines, indices):
            np.testing.assert_array_equal(line.get_ydata(), y[index])
    for ax in list(axes.flat)[5:]:
        assert not ax.get_visible()
        assert not ax.lines
        assert not ax.collections
        assert ax.get_title() == ""


@pytest.mark.parametrize("as_series", [False, True])
def test_wrapped_facets_preserve_categorical_order_and_missing_membership(as_series):
    col = pd.Categorical(
        ["A", None, "B", "Missing"], categories=["B", "unused", "Missing", "A"]
    )
    if as_series:
        col = pd.Series(col, name="Group")

    _, axes = plot_spectra(np.arange(8).reshape(4, 2), col=col, col_wrap=3)

    assert axes.shape == (2, 3)
    prefix = "Group = " if as_series else ""
    assert [ax.get_title() for ax in list(axes.flat)[:4]] == [
        prefix + label for label in ["B", "Missing", "A", "Missing"]
    ]
    for ax, index in zip(axes.flat, [2, 3, 0, 1]):
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), [index * 2, index * 2 + 1]
        )
    assert [ax.get_visible() for ax in axes.flat] == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_wrapped_facets_keep_all_missing_membership_in_one_visible_panel():
    _, axes = plot_spectra([[1, 2], [3, 4]], col=[None, pd.NA], col_wrap=2)

    assert axes.shape == (1, 2)
    assert axes[0, 0].get_title() == "Missing"
    assert len(axes[0, 0].lines) == 2
    assert not axes[0, 1].get_visible()


@pytest.mark.parametrize("sharex", [True, False, "row", "col"])
@pytest.mark.parametrize("sharey", [True, False, "row", "col"])
def test_wrapped_facets_share_axes_by_physical_grid_positions(sharex, sharey):
    _, axes = plot_spectra(
        np.arange(10).reshape(5, 2),
        col=list("ABCDE"),
        col_wrap=2,
        sharex=sharex,
        sharey=sharey,
    )
    visible = [
        (position, ax) for position, ax in np.ndenumerate(axes) if ax.get_visible()
    ]

    for coordinate, sharing in [("x", sharex), ("y", sharey)]:
        for (r1, c1), first in visible:
            for (r2, c2), second in visible:
                if first is second:
                    continue
                expected = (
                    sharing is True
                    or (sharing == "row" and r1 == r2)
                    or (sharing == "col" and c1 == c2)
                )
                group = getattr(first, f"get_shared_{coordinate}_axes")()
                assert group.joined(first, second) == expected


@pytest.mark.parametrize("sharex", [True, False, "row", "col"])
def test_wrapped_facets_keep_tick_labels_at_visible_bottom_edges(sharex):
    fig, axes = plot_spectra(
        np.arange(6).reshape(3, 2), col=list("ABC"), col_wrap=2, sharex=sharex
    )
    fig.canvas.draw()

    for position in [(0, 1), (1, 0)]:
        labels = axes[position].get_xticklabels()
        assert labels
        assert all(label.get_visible() for label in labels)
    assert bool(axes[0, 0].get_xticklabels()) == (sharex is False or sharex == "row")
    assert not axes[1, 1].get_visible()


@pytest.mark.parametrize("sharey", [True, False, "row", "col"])
def test_wrapped_facets_apply_symmetry_after_all_shared_data_and_bands_are_drawn(
    sharey,
):
    y = np.array([[1, 2], [10, 20], [-300, -200], [1000, 2000], [400, 500]])
    options = {
        "col": list("ABCDE"),
        "col_wrap": 2,
        "sharey": sharey,
        "ymin": y - 5,
        "ymax": y + 50,
    }
    _, reference = plot_spectra(y, **options)
    _, axes = plot_spectra(y, symmetric=True, **options)

    for ax, other in zip(axes.flat, reference.flat):
        if not ax.get_visible():
            continue
        limit = max(abs(value) for value in other.get_ylim())
        np.testing.assert_allclose(ax.get_ylim(), [-limit, limit])


@pytest.mark.parametrize(
    "col_wrap", [0, -1, 1.5, 2.0, "2", True, False, np.bool_(True), [2]]
)
def test_wrapped_facets_reject_invalid_width_without_creating_figures(col_wrap):
    figures = plt.get_fignums()

    with pytest.raises(ValueError, match="col_wrap.*positive integer"):
        plot_spectra([[1, 2], [3, 4]], col=["A", "B"], col_wrap=col_wrap)

    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "col_wrap.*col"),
        ({"row": ["A", "B"]}, "col_wrap.*row"),
        ({"row": ["A", "B"], "col": ["X", "Y"]}, "col_wrap.*row"),
    ],
)
def test_wrapped_facets_require_col_without_row(options, message):
    figures = plt.get_fignums()

    with pytest.raises(ValueError, match=message):
        plot_spectra([[1, 2], [3, 4]], col_wrap=2, **options)

    assert plt.get_fignums() == figures


def test_wrapped_facets_reject_supplied_axis_without_modifying_it():
    fig, ax = plt.subplots()
    existing = ax.plot([0, 1], [1, 2])[0]
    figures = plt.get_fignums()

    with pytest.raises(ValueError, match="ax.*facet"):
        plot_spectra([[1, 2], [3, 4]], col=["A", "B"], col_wrap=1, ax=ax)

    assert plt.get_fignums() == figures
    assert fig.axes == [ax]
    assert list(ax.lines) == [existing]
