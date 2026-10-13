import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

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
