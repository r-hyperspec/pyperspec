"""X breaks preserve axes and data coordinates while omitting selected intervals."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from pyspc.plotting import break_xaxis, plot_spectra


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


def test_break_xaxis_keeps_axes_data_labels_and_legend():
    fig, ax = plt.subplots()
    x, y = [0, 10], [0, 10]
    line = ax.plot(x, y, label="Spectrum")[0]
    ax.set(
        xlim=(0, 10),
        ylim=(0, 10),
        xlabel="Wavenumber",
        ylabel="Intensity",
        title="Sample",
    )
    legend = ax.legend()

    returned = break_xaxis(ax, [(4, 8)])

    assert returned is ax
    assert fig.axes == [ax]
    assert ax.lines[0] is line
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    np.testing.assert_allclose(ax.get_xlim(), [0, 10])
    assert ax.get_xlabel() == "Wavenumber"
    assert ax.get_ylabel() == "Intensity"
    assert ax.get_title() == "Sample"
    assert ax.get_legend() is legend
    display = ax.transData.transform([[0, 0], [4, 0], [8, 0], [10, 0]])[:, 0]
    np.testing.assert_allclose(
        (display[1] - display[0]) / 4, (display[3] - display[2]) / 2
    )
    assert 0 < display[2] - display[1] < ax.bbox.width * 0.1
    points = np.array([[1, 2], [3, 4], [9, 8]])
    np.testing.assert_allclose(
        ax.transData.inverted().transform(ax.transData.transform(points)), points
    )


def test_break_xaxis_interpolates_sparse_lines_and_bands_at_break_boundaries():
    _, ax = plt.subplots()
    line = ax.plot([0, 10], [0, 10])[0]
    band = ax.fill_between([0, 10], [-1, 9], [1, 11])
    ax.set(xlim=(0, 10), ylim=(-2, 12))

    break_xaxis(ax, [(4, 8)])

    for artist, path in [(line, line.get_path()), (band, band.get_paths()[0])]:
        rendered = artist.get_transform().transform_path(path)
        data = ax.transData.inverted().transform(rendered.vertices)
        for boundary in [4, 8]:
            values = data[np.isclose(data[:, 0], boundary), 1]
            assert len(values)
            if artist is line:
                np.testing.assert_allclose(values, boundary)
            else:
                np.testing.assert_allclose(
                    np.sort(values), [boundary - 1, boundary + 1]
                )
        clipping = artist.get_clip_path().get_fully_transformed_path()
        points = ax.transData.transform([[2, 5], [6, 5], [9, 5]])
        np.testing.assert_array_equal(
            clipping.contains_points(points), [True, False, True]
        )


def test_break_xaxis_leaves_a_visible_gap_in_rendered_lines_and_bands():
    fig, ax = plt.subplots()
    ax.plot([0, 10], [3, 3], color="red", linewidth=4)
    ax.fill_between([0, 10], [0, 0], [6, 6], color="red")
    ax.set(xlim=(0, 10), ylim=(0, 6))
    break_xaxis(ax, [(4, 8)])
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())

    for x, color in [(2, "red"), (6, ax.get_facecolor()), (9, "red")]:
        px, py = np.round(ax.transData.transform((x, 3))).astype(int)
        np.testing.assert_allclose(
            pixels[pixels.shape[0] - 1 - py, px] / 255, to_rgba(color)
        )


@pytest.mark.parametrize("inverted", [False, True])
def test_break_xaxis_uses_original_coordinates_for_multiple_breaks_and_ticks(inverted):
    _, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])
    ax.set_xlim((10, 0) if inverted else (0, 10))

    break_xaxis(ax, [(7, 9), (2, 4)])

    np.testing.assert_allclose(ax.get_xlim(), [10, 0] if inverted else [0, 10])
    ticks = ax.get_xticks()
    assert len(ticks) >= 4
    assert not np.any(((ticks > 2) & (ticks < 4)) | ((ticks > 7) & (ticks < 9)))
    points = np.array([[1, 0], [5, 0], [9.5, 0]])
    np.testing.assert_allclose(
        ax.transData.inverted().transform(ax.transData.transform(points)), points
    )


def test_break_xaxis_can_be_applied_to_each_shared_axis_and_reapplied():
    _, axes = plt.subplots(2, 1, sharex=True)
    for ax in axes:
        ax.plot([0, 10], [0, 1])
    axes[0].set_xlim(0, 10)

    for ax in axes:
        break_xaxis(ax, [(4, 8)])
    counts = [len(ax.lines) for ax in axes]
    for ax in axes:
        break_xaxis(ax, [(4, 8)])

    assert [len(ax.lines) for ax in axes] == counts
    positions = []
    for ax in axes:
        display = ax.transData.transform([[4, 0], [8, 0]])[:, 0]
        positions.append((display - ax.bbox.x0) / ax.bbox.width)
    np.testing.assert_allclose(*positions)


@pytest.mark.parametrize("x_breaks", [[], [(20, 30)]])
def test_break_xaxis_is_a_noop_without_visible_breaks(x_breaks):
    _, ax = plt.subplots()
    original = ax.plot([0, 10], [0, 1])[0]
    ax.set_xlim(0, 10)

    assert break_xaxis(ax, x_breaks) is ax
    assert ax.get_xscale() == "linear"
    assert list(ax.lines) == [original]
    assert original.get_clip_path() is None


@pytest.mark.parametrize(
    "x_breaks, error",
    [
        ([2, 4], ValueError),
        ([[2, 3, 4]], ValueError),
        ([(4, 2)], ValueError),
        ([(2, 2)], ValueError),
        ([(2, np.inf)], ValueError),
        ([(np.nan, 4)], ValueError),
        ([(2, 5), (4, 6)], ValueError),
        ([(2, 4), (4, 6)], ValueError),
        ([["a", "b"]], TypeError),
        ([(False, True)], TypeError),
    ],
)
def test_invalid_breaks_are_rejected_before_axes_are_modified(x_breaks, error):
    fig, ax = plt.subplots()
    line = ax.plot([0, 10], [0, 1])[0]
    ax.set_xlim(0, 10)

    with pytest.raises(error, match="x_breaks"):
        break_xaxis(ax, x_breaks)

    assert ax.get_xscale() == "linear"
    assert list(ax.lines) == [line]
    assert fig.axes == [ax]


def test_break_xaxis_rejects_omitting_the_entire_view():
    _, ax = plt.subplots()
    ax.set_xlim(0, 10)
    with pytest.raises(ValueError, match="entire"):
        break_xaxis(ax, [(-1, 11)])
    assert ax.get_xscale() == "linear"


def test_break_xaxis_rejects_non_linear_scales_without_modifying_artists():
    _, ax = plt.subplots()
    line = ax.plot([1, 10], [0, 1])[0]
    ax.set_xscale("log")
    with pytest.raises(ValueError, match="linear"):
        break_xaxis(ax, [(2, 4)])
    assert ax.get_xscale() == "log"
    assert list(ax.lines) == [line]


@pytest.mark.parametrize(
    "layout",
    [
        {},
        {"row": ["A", "A", "B", "B", "A"], "col": ["X", "Y", "X", "Y", "X"]},
        {"col": list("ABCDE"), "col_wrap": 3},
    ],
)
@pytest.mark.parametrize("hue", [[1, 2, 3, 4, 5], ["A", "B", "A", "B", "A"]])
def test_plot_spectra_applies_breaks_to_spectral_panels_preserving_hue_guides(
    layout, hue
):
    x = np.linspace(0, 10, 11)
    y = np.tile(x, (5, 1))
    fig, axes = plot_spectra(
        y, x=x, ymin=y - 1, ymax=y + 1, hue=hue, x_breaks=[(4, 8)], **layout
    )
    panels = np.asarray(axes).reshape(-1)

    assert fig.axes[: panels.size] == list(panels)
    for ax in panels:
        if ax.get_visible():
            ticks = ax.get_xticks()
            assert not np.any((ticks > 4) & (ticks < 8))
            assert ax.get_xscale() != "linear"
        else:
            assert not ax.lines
    if isinstance(hue[0], int):
        assert len(fig.axes) == panels.size + 1
        assert fig.axes[-1].get_xscale() == "linear"
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), [1, 5])
    else:
        assert len(fig.axes) == panels.size
        legend = fig.legends[0] if layout else panels[0].get_legend()
        assert [label.get_text() for label in legend.get_texts()] == ["A", "B"]


def test_plot_spectra_validates_breaks_before_creating_a_figure():
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="x_breaks"):
        plot_spectra([[1, 2]], x_breaks=[(4, 2)])
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("x_breaks", [[(-2, 2)], [(8, 12)], [(-2, 2), (8, 12)]])
def test_break_xaxis_clips_partial_intervals_to_the_current_view(x_breaks):
    _, ax = plt.subplots()
    line = ax.plot([0, 10], [0, 1])[0]
    ax.set(xlim=(0, 10), ylim=(0, 1))

    break_xaxis(ax, x_breaks)

    np.testing.assert_allclose(ax.get_xlim(), [0, 10])
    clipping = line.get_clip_path().get_fully_transformed_path()
    points = ax.transData.transform([[1, 0.5], [5, 0.5], [9, 0.5]])
    np.testing.assert_array_equal(
        clipping.contains_points(points),
        [x_breaks[0][0] >= 0, True, x_breaks[-1][1] <= 10],
    )


def test_break_xaxis_preserves_existing_nan_gaps_in_lines_and_bands():
    fig, ax = plt.subplots()
    x = np.arange(7)
    y = np.array([2, 2, np.nan, 2, 2, 2, 2])
    line = ax.plot(x, y, color="red", linewidth=4)[0]
    ax.fill_between(x, y - 1, y + 1, color="red")
    ax.set(xlim=(0, 6), ylim=(0, 4))
    break_xaxis(ax, [(4, 5)])
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())

    np.testing.assert_array_equal(line.get_ydata(), y)
    px, py = np.round(ax.transData.transform((2, 2))).astype(int)
    np.testing.assert_allclose(
        pixels[pixels.shape[0] - 1 - py, px] / 255, ax.get_facecolor()
    )


def test_break_xaxis_reapplies_after_limits_change_without_distorting_paths():
    _, ax = plt.subplots()
    line = ax.plot([0, 20], [0, 20])[0]
    ax.set_xlim(0, 10)
    break_xaxis(ax, [(4, 8)])
    ax.set_xlim(0, 20)

    break_xaxis(ax, [(4, 8), (12, 16)])

    rendered = line.get_transform().transform_path(line.get_path())
    data = ax.transData.inverted().transform(rendered.vertices)
    np.testing.assert_allclose(data[:, 0], [0, 4, 8, 12, 16, 20])
    np.testing.assert_allclose(data[:, 1], data[:, 0])


def test_break_xaxis_rejects_invalid_axis():
    with pytest.raises(TypeError, match="ax"):
        break_xaxis("not an axis", [(4, 8)])


def test_break_xaxis_keeps_tick_labels_separated_across_compressed_gaps():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1000], [0, 1])
    ax.set_xlim(0, 1000)
    break_xaxis(ax, [(400, 800)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]

    assert all(first.x1 < second.x0 for first, second in zip(labels, labels[1:]))


def test_break_xaxis_clips_the_horizontal_spines():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 10), ylim=(0, 1))
    break_xaxis(ax, [(4, 8)])
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    px, py = np.round(ax.transData.transform((6, 0))).astype(int)

    np.testing.assert_allclose(
        pixels[pixels.shape[0] - 1 - py, px] / 255, ax.get_facecolor()
    )


def test_plot_spectra_rejects_non_linear_breaks_before_drawing():
    fig, ax = plt.subplots()
    existing = ax.plot([1, 10], [0, 1])[0]
    ax.set_xscale("log")

    with pytest.raises(ValueError, match="linear"):
        plot_spectra([[1, 2]], x=[1, 10], ax=ax, x_breaks=[(2, 4)])

    assert list(ax.lines) == [existing]
    assert fig.axes == [ax]
    assert ax.get_xscale() == "log"


@pytest.mark.parametrize(
    "gap, major, minor, expected",
    [
        (50, 100, 20, 50),
        (None, 100, 20, 20),
        (None, 100, None, 50),
        (None, None, 20, 20),
    ],
)
def test_break_gap_is_measured_in_retained_x_data_units(gap, major, minor, expected):
    _, ax = plt.subplots()
    ax.plot([0, 1000], [0, 1])
    ax.set_xlim(0, 1000)

    break_xaxis(
        ax, [(300, 800)], x_breaks_gap=gap, xtick_major=major, xtick_minor=minor
    )

    transformed = ax.xaxis.get_transform().transform(np.array([100, 200, 300, 800]))
    np.testing.assert_allclose(transformed[1] - transformed[0], 100)
    np.testing.assert_allclose(transformed[3] - transformed[2], expected)


def test_default_gap_is_half_the_automatic_major_spacing():
    _, ax = plt.subplots()
    ax.plot([0, 1000], [0, 1])
    ax.set_xlim(0, 1000)

    break_xaxis(ax, [(300, 800)])

    major = np.min(np.diff(ax.get_xticks()))
    transformed = ax.xaxis.get_transform().transform(np.array([300, 800]))
    np.testing.assert_allclose(np.diff(transformed), major / 2)


@pytest.mark.parametrize("inverted", [False, True])
def test_explicit_major_and_minor_spacing_omits_only_ticks_inside_breaks(inverted):
    _, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1000], [0, 1])
    limits = [1000, 0] if inverted else [0, 1000]
    ax.set_xlim(limits)

    break_xaxis(ax, [(300, 800)], xtick_major=100, xtick_minor=20)

    np.testing.assert_array_equal(ax.get_xticks(), [0, 100, 200, 300, 800, 900, 1000])
    np.testing.assert_array_equal(
        ax.get_xticks(minor=True),
        [
            20,
            40,
            60,
            80,
            120,
            140,
            160,
            180,
            220,
            240,
            260,
            280,
            820,
            840,
            860,
            880,
            920,
            940,
            960,
            980,
        ],
    )
    np.testing.assert_allclose(ax.get_xlim(), limits)


def test_plot_spectra_can_configure_major_and_minor_ticks_without_breaks():
    fig, ax = plot_spectra([[1, 2]], x=[0, 1000], xtick_major=100, xtick_minor=20)
    ax.set_xlim(0, 1000)

    np.testing.assert_array_equal(ax.get_xticks()[1:-1], np.arange(0, 1001, 100))
    minor = ax.get_xticks(minor=True)
    minor = minor[(minor > 0) & (minor < 1000)]
    np.testing.assert_array_equal(
        minor, [value for value in range(20, 1000, 20) if value % 100]
    )
    assert ax.get_xscale() == "linear"
    assert len(ax.lines) == 1
    assert fig.axes == [ax]


def test_plot_spectra_applies_tick_and_gap_controls_to_wrapped_facets_only():
    x = np.linspace(0, 1000, 11)
    fig, axes = plot_spectra(
        np.tile(x, (3, 1)),
        x=x,
        col=list("ABC"),
        col_wrap=2,
        hue=[1, 2, 3],
        x_breaks=[(300, 800)],
        x_breaks_gap=50,
        xtick_major=100,
        xtick_minor=20,
    )

    for ax in axes.flat:
        if not ax.get_visible():
            assert not ax.lines
            continue
        ticks = ax.get_xticks()
        assert np.all(ticks % 100 == 0)
        assert not np.any((ticks > 300) & (ticks < 800))
        minor = ax.get_xticks(minor=True)
        assert len(minor)
        assert np.all(minor % 20 == 0)
        assert not np.any((minor > 300) & (minor < 800))
        transformed = ax.xaxis.get_transform().transform(np.array([300, 800]))
        np.testing.assert_allclose(np.diff(transformed), 50)
    assert fig.axes[-1].get_xscale() == "linear"
    np.testing.assert_allclose(fig.axes[-1].get_ylim(), [1, 3])


@pytest.mark.parametrize("option", ["x_breaks_gap", "xtick_major", "xtick_minor"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "20"])
def test_invalid_tick_and_gap_options_fail_before_drawing(option, value):
    figures = plt.get_fignums()

    with pytest.raises((TypeError, ValueError), match=option):
        plot_spectra([[1, 2]], **{option: value})

    assert plt.get_fignums() == figures
