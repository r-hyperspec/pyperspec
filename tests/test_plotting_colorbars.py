"""Continuous hue guides use the same scale as the spectral artists."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_rgba

from pyspc.plotting import plot_spectra


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


@pytest.fixture(params=["single", "rectangular", "wrapped"])
def layout(request):
    if request.param == "rectangular":
        return {"row": ["A", "A", "B", "B", "A"], "col": ["X", "X", "Y", "X", "Y"]}
    if request.param == "wrapped":
        return {"col": ["A", "B", "C", "D", None], "col_wrap": 3}
    return {}


@pytest.mark.parametrize("palette", [None, "plasma"])
@pytest.mark.parametrize("hue_name", [None, "Dose"])
def test_colorbar_matches_global_hue_scale_and_band_colors(layout, palette, hue_name):
    y = np.arange(10).reshape(5, 2)
    hue = pd.Series([10, 20, 30, np.nan, 10], name=hue_name, index=[5, 4, 3, 2, 1])
    fig, axes = plot_spectra(
        y, hue=hue, palette=palette, ymin=y - 1, ymax=y + 2, **layout
    )
    panels = np.asarray(axes).reshape(-1)
    visible = [ax for ax in panels if ax.get_visible()]

    assert fig.axes[:-1] == list(panels)
    bar_ax = fig.axes[-1]
    assert bar_ax not in panels
    assert bar_ax.get_ylabel() == ("" if hue_name is None else hue_name)
    np.testing.assert_allclose(bar_ax.get_ylim(), [10, 30])
    assert not fig.legends
    assert all(ax.get_legend() is None for ax in panels)
    cmap = plt.get_cmap("viridis" if palette is None else palette)
    expected = np.array([cmap(0.0), cmap(0.5), cmap(1.0), to_rgba("gray"), cmap(0.0)])
    for ax in visible:
        for line, band in zip(ax.lines, ax.collections):
            index = int(line.get_ydata()[0]) // 2
            np.testing.assert_allclose(to_rgba(line.get_color()), expected[index])
            np.testing.assert_allclose(
                band.get_facecolor(), [to_rgba(expected[index], 0.3)]
            )

    # Inspect the rendered guide, including the space reserved beside the panels.
    fig.canvas.draw()
    mesh = next(artist for artist in bar_ax.collections if isinstance(artist, QuadMesh))
    np.testing.assert_allclose(mesh.get_clim(), [10, 30])
    np.testing.assert_allclose(mesh.to_rgba([10, 20, 30]), expected[:3])
    renderer = fig.canvas.get_renderer()
    bar_box = bar_ax.get_window_extent(renderer)
    assert bar_box.x0 >= max(ax.get_window_extent(renderer).x1 for ax in visible)
    np.testing.assert_allclose(
        [bar_box.y0, bar_box.y1],
        [
            min(ax.get_window_extent(renderer).y0 for ax in visible),
            max(ax.get_window_extent(renderer).y1 for ax in visible),
        ],
        atol=0.1,
    )
    figure_box = fig.get_window_extent(renderer)
    label_box = bar_ax.get_tightbbox(renderer)
    assert figure_box.contains(label_box.x0, label_box.y0)
    assert figure_box.contains(label_box.x1, label_box.y1)


def test_legend_false_omits_colorbar_without_changing_hue_colors(layout):
    y = np.arange(10).reshape(5, 2)
    hue = [10, 20, 30, np.nan, 10]
    fig, axes = plot_spectra(y, hue=hue, legend=False, **layout)
    panels = np.asarray(axes).reshape(-1)

    assert fig.axes == list(panels)
    assert not fig.legends
    cmap = plt.get_cmap("viridis")
    expected = [cmap(0.0), cmap(0.5), cmap(1.0), to_rgba("gray"), cmap(0.0)]
    for ax in panels:
        for line in ax.lines:
            index = int(line.get_ydata()[0]) // 2
            np.testing.assert_allclose(to_rgba(line.get_color()), expected[index])


def test_all_missing_numeric_hue_has_no_colorbar_or_legend(layout):
    hue = pd.Series([pd.NA] * 5, dtype="Float64", name="Dose")
    fig, axes = plot_spectra(np.ones((5, 2)), hue=hue, **layout)
    panels = np.asarray(axes).reshape(-1)

    assert fig.axes == list(panels)
    assert not fig.legends
    for ax in panels:
        assert ax.get_legend() is None
        for line in ax.lines:
            np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba("gray"))


def test_supplied_axis_gets_colorbar_without_changing_other_axes():
    fig, (target, other) = plt.subplots(1, 2)
    existing = other.plot([0, 1], [5, 10], label="Existing")[0]
    other_legend = other.legend()
    other_position = other.get_position().bounds

    returned_fig, returned_ax = plot_spectra(
        [[1, 2], [3, 4]], hue=pd.Series([10, 20], name="Dose"), ax=target
    )

    assert returned_fig is fig
    assert returned_ax is target
    assert fig.axes[:-1] == [target, other]
    assert fig.axes[-1].get_ylabel() == "Dose"
    assert list(other.lines) == [existing]
    assert other.get_legend() is other_legend
    np.testing.assert_allclose(other.get_position().bounds, other_position)


def test_legend_false_overlay_preserves_existing_colorbar():
    fig, ax = plot_spectra([[1, 2], [3, 4]], hue=pd.Series([10, 20], name="Dose"))
    original_axes = list(fig.axes)
    bar_ax = original_axes[-1]
    original_limits = bar_ax.get_ylim()

    plot_spectra([[5, 6]], hue=[100], ax=ax, legend=False)

    assert fig.axes == original_axes
    assert len(ax.lines) == 3
    assert bar_ax.get_ylabel() == "Dose"
    np.testing.assert_allclose(bar_ax.get_ylim(), original_limits)
