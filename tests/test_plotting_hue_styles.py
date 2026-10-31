"""Discrete hue styles apply consistently to spectra and their legend handles."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from pyspc import SpectraFrame
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
        return {
            "row": ["r1", "r1", "r2", "r2", "r1", "r2"],
            "col": ["x", "y", "x", "y", "x", "y"],
        }
    if request.param == "wrapped":
        return {"col": ["z", "x", "z", "y", None, "x"], "col_wrap": 3}
    return {}


@pytest.mark.parametrize("legend", [True, False])
def test_hue_styles_override_global_defaults_and_match_legends_and_palette(
    layout, legend
):
    y = np.arange(18).reshape(6, 3)
    hue = pd.Series(
        pd.Categorical(["B", "A", "C", "B", None, "A"], categories=["C", "B", "A"]),
        name="Treatment",
        index=range(6, 0, -1),
    )
    styles = {
        "A": {"linestyle": "--"},
        "B": {"lw": 3, "alpha": 0.4},
        "unused": {"linewidth": 9},
    }
    original = deepcopy(styles)

    fig, axes = plot_spectra(
        y,
        hue=hue,
        hue_styles=styles,
        palette={"A": "red", "B": "blue", "C": "green"},
        lw=1.25,
        ls=":",
        alpha=0.9,
        ymin=y - 1,
        ymax=y + 1,
        fill_kws={"alpha": 0.2},
        legend=legend,
        **layout,
    )

    # Missing hue retains global styles and stays separate from named categories.
    expected = {
        "A": (1.25, "--", 0.9, "red"),
        "B": (3, ":", 0.4, "blue"),
        "C": (1.25, ":", 0.9, "green"),
        "Missing": (1.25, ":", 0.9, "gray"),
    }
    for ax in np.atleast_1d(axes).flat:
        for line, ribbon in zip(ax.lines, ax.collections):
            index = int(line.get_ydata()[0]) // 3
            group = "Missing" if pd.isna(hue.iloc[index]) else hue.iloc[index]
            width, style, alpha, color = expected[group]
            assert line.get_linewidth() == width
            assert line.get_linestyle() == style
            assert line.get_alpha() == alpha
            assert to_rgba(line.get_color()) == to_rgba(color)
            np.testing.assert_allclose(ribbon.get_facecolor(), [to_rgba(color, 0.2)])
            np.testing.assert_array_equal(ribbon.get_linewidths(), [0])

    guide = fig.legends[0] if layout and legend else fig.axes[0].get_legend()
    if legend:
        assert guide.get_title().get_text() == "Treatment"
        assert [text.get_text() for text in guide.get_texts()] == [
            "C",
            "B",
            "A",
            "Missing",
        ]
        for handle, group in zip(guide.legend_handles, ["C", "B", "A", "Missing"]):
            width, style, alpha, color = expected[group]
            assert handle.get_linewidth() == width
            assert handle.get_linestyle() == style
            assert handle.get_alpha() == alpha
            assert to_rgba(handle.get_color()) == to_rgba(color)
    else:
        assert guide is None
        assert not fig.legends
    assert styles == original


@pytest.mark.parametrize(
    "hue, color_type",
    [([20, 10, 20], "discrete"), (pd.Categorical([20, 10, 20]), "auto")],
)
def test_hue_styles_accept_numeric_categories(hue, color_type):
    _, ax = plot_spectra(
        np.ones((3, 2)),
        hue=hue,
        color_type=color_type,
        hue_styles={10: {"ls": "--"}, 20: {"linewidth": 4}},
        linewidth=1,
        linestyle="-",
    )

    assert [line.get_linewidth() for line in ax.lines] == [4, 1, 4]
    assert [line.get_linestyle() for line in ax.lines] == ["-", "--", "-"]
    assert [handle.get_linewidth() for handle in ax.get_legend().legend_handles] == [
        1,
        4,
    ]


def test_missing_hue_retains_defaults_separately_from_literal_missing_category():
    _, ax = plot_spectra(
        [[1, 2], [3, 4]],
        hue=["Missing", None],
        hue_styles={"Missing": {"linewidth": 3}},
        linewidth=1,
    )

    assert [line.get_linewidth() for line in ax.lines] == [3, 1]
    assert [handle.get_linewidth() for handle in ax.get_legend().legend_handles] == [
        3,
        1,
    ]


@pytest.mark.parametrize("faceted", [False, True])
@pytest.mark.parametrize(
    "options, error",
    [
        ({"hue_styles": {}}, ValueError),
        ({"hue": [10, 20], "hue_styles": {}}, ValueError),
        ({"hue": pd.Series([np.nan, np.nan]), "hue_styles": {}}, ValueError),
        ({"hue": ["A", "B"], "hue_styles": []}, TypeError),
        ({"hue": ["A", "B"], "hue_styles": {"A": []}}, TypeError),
        ({"hue": ["A", "B"], "hue_styles": {"A": {"color": "red"}}}, ValueError),
        ({"hue": ["A", "B"], "hue_styles": {"A": {"alpha": 1.5}}}, ValueError),
        ({"hue": ["A", "B"], "hue_styles": {"A": {"linestyle": "bad"}}}, ValueError),
        ({"hue": ["A", "B"], "hue_styles": {"A": {"linewidth": "bad"}}}, ValueError),
    ],
)
def test_hue_styles_reject_invalid_options_before_modifying_axes(
    faceted, options, error
):
    _, ax = plt.subplots()
    existing = ax.plot([0, 1], [2, 3])[0]
    figures = plt.get_fignums()
    layout = {"col": ["X", "Y"]} if faceted else {"ax": ax}

    with pytest.raises(error):
        plot_spectra([[1, 2], [3, 4]], **options, **layout)

    assert plt.get_fignums() == figures
    assert list(ax.lines) == [existing]


@pytest.mark.parametrize(
    "method", ["plot", "plot_mean_sd", "plot_quantiles", "plot_aggregate_with_ribbon"]
)
def test_frame_plotting_methods_forward_hue_styles(method):
    sf = SpectraFrame(
        np.arange(24).reshape(8, 3),
        data=pd.DataFrame(
            {
                "batch": ["B", "A"] * 4,
                "treatment": ["control"] * 4 + ["treated"] * 4,
            }
        ),
    )
    args = (
        (lambda spectra: np.quantile(spectra, [0.25, 0.5, 0.75], axis=0),)
        if method == "plot_aggregate_with_ribbon"
        else ()
    )

    _, axes = getattr(sf, method)(
        *args,
        col="batch",
        hue="treatment",
        hue_styles={"control": {"linewidth": 2.5, "linestyle": "--"}},
        palette={"control": "blue", "treated": "red"},
        linewidth=1,
        linestyle="-",
    )

    for ax in axes.flat:
        assert len(ax.lines) == (4 if method == "plot" else 2)
        for line in ax.lines:
            control = to_rgba(line.get_color()) == to_rgba("blue")
            assert line.get_linewidth() == (2.5 if control else 1)
            assert line.get_linestyle() == ("--" if control else "-")
