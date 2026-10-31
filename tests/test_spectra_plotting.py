"""SpectraFrame resolves metadata and delegates rendering to generic plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from pyspc import SpectraFrame


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


@pytest.fixture
def frame():
    return SpectraFrame(
        np.arange(12).reshape(4, 3),
        wl=[400, 600, 800],
        data=pd.DataFrame(
            {
                "batch": pd.Categorical(
                    ["B", "A", "B", "A"], categories=["B", "A", "unused"]
                ),
                "treatment": ["control", "control", "treated", "treated"],
                "dose": [10, 20, 30, 40],
            },
            index=pd.Index(["s3", "s1", "s4", "s2"], name="sample"),
        ),
    )


def test_frame_plot_uses_spectra_and_wavelengths_without_aggregation(frame):
    fig, ax = frame.plot()

    assert isinstance(ax, Axes)
    assert fig.axes == [ax]
    assert len(ax.lines) == frame.nspc
    for line, spectrum in zip(ax.lines, frame.spc):
        np.testing.assert_array_equal(line.get_xdata(), frame.wl)
        np.testing.assert_array_equal(line.get_ydata(), spectrum)
    assert ax.get_legend() is None


def test_frame_plot_resolves_named_metadata_and_numeric_hue(frame):
    fig, axes = frame.plot(row="batch", col="treatment", hue="dose", palette="plasma")

    assert axes.shape == (2, 2)
    assert [ax.get_title() for ax in axes.flat] == [
        "batch = B | treatment = control",
        "batch = B | treatment = treated",
        "batch = A | treatment = control",
        "batch = A | treatment = treated",
    ]
    for ax, index in zip(axes.flat, [0, 2, 1, 3]):
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), frame.spc[index])
        np.testing.assert_allclose(
            to_rgba(ax.lines[0].get_color()), plt.get_cmap("plasma")(index / 3)
        )
    assert fig.axes[-1].get_ylabel() == "dose"
    np.testing.assert_allclose(fig.axes[-1].get_ylim(), [10, 40])


def test_frame_plot_external_series_align_by_position_and_preserve_categories(frame):
    groups = pd.Series(
        pd.Categorical(["X", "Y", "X", "Y"], categories=["Y", "X"]),
        index=["s2", "s4", "s1", "s3"],
        name="External",
    )
    original = groups.copy()

    _, axes = frame.plot(
        row=groups,
        hue=np.array(["blue", "red", "blue", "red"]),
        palette={"blue": "blue", "red": "red"},
    )

    assert [ax.get_title() for ax in axes.flat] == ["External = Y", "External = X"]
    for ax, indices, color in zip(axes.flat, [[1, 3], [0, 2]], ["red", "blue"]):
        for line, index in zip(ax.lines, indices):
            np.testing.assert_array_equal(line.get_ydata(), frame.spc[index])
            assert to_rgba(line.get_color()) == to_rgba(color)
    pd.testing.assert_series_equal(groups, original)


def test_frame_plot_accepts_index_as_membership(frame):
    _, axes = frame.plot(row="index")

    assert axes.shape == (4, 1)
    for ax, label, index in zip(axes.flat, ["s1", "s2", "s3", "s4"], [1, 3, 0, 2]):
        assert ax.get_title() == f"sample = {label}"
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), frame.spc[index])


def test_frame_plot_preserves_missing_membership_and_literal_na(frame):
    frame.data["group"] = [None, "Missing", "NA", np.nan]

    _, axes = frame.plot(col="group", hue="group")

    assert [ax.get_title() for ax in axes.flat] == [
        "group = Missing",
        "group = NA",
        "group = Missing",
    ]
    assert [len(ax.lines) for ax in axes.flat] == [1, 1, 2]
    for line in axes[0, 2].lines:
        assert to_rgba(line.get_color()) == to_rgba("gray")


def test_frame_plot_forwards_bands_wrapping_titles_ticks_and_breaks(frame):
    before = frame.data.copy(deep=True)
    spectra = frame.spc.copy()

    _, axes = frame.plot(
        col="batch",
        col_wrap=1,
        ymin=frame.spc - 1,
        ymax=frame.spc + 2,
        title_formatter=lambda row_value, col_value: f"Batch {col_value}",
        x_breaks=[(500, 700)],
        xtick_major=100,
        xtick_minor=20,
        x_breaks_gap=50,
        linewidth=3,
        color="black",
    )

    assert axes.shape == (2, 1)
    assert [ax.get_title() for ax in axes.flat] == ["Batch B", "Batch A"]
    for ax in axes.flat:
        assert len(ax.collections) == 2
        assert ax.lines[0].get_linewidth() == 3
        assert to_rgba(ax.lines[0].get_color()) == to_rgba("black")
        assert ax.get_xscale() != "linear"
        ticks = ax.get_xticks()
        assert not np.any((ticks > 500) & (ticks < 700))
    pd.testing.assert_frame_equal(frame.data, before)
    np.testing.assert_array_equal(frame.spc, spectra)


def test_frame_plot_uses_supplied_single_axis(frame):
    fig, (ax, other) = plt.subplots(1, 2)
    existing = ax.plot([400, 800], [1, 2])[0]

    returned_fig, returned_ax = frame.plot(ax=ax)

    assert returned_fig is fig
    assert returned_ax is ax
    assert list(ax.lines)[0] is existing
    assert len(ax.lines) == frame.nspc + 1
    assert not other.lines


@pytest.mark.parametrize("role", ["row", "col", "hue"])
@pytest.mark.parametrize(
    "value, error",
    [("unknown", KeyError), ([1, 2], ValueError), ([[1], [2], [3], [4]], ValueError)],
)
def test_frame_plot_rejects_invalid_metadata_before_plotting(frame, role, value, error):
    figures = plt.get_fignums()

    with pytest.raises(error):
        frame.plot(**{role: value})

    assert plt.get_fignums() == figures
