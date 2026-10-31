"""Aggregation plots summarize spectra before using the generic renderer."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from pyspc import SpectraFrame


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


@pytest.fixture(params=["plot_mean_sd", "plot_quantiles"])
def method(request):
    return request.param


@pytest.fixture
def frame():
    return SpectraFrame(
        np.arange(24, dtype=float).reshape(8, 3) ** 2,
        wl=[400, 600, 800],
        data=pd.DataFrame(
            {
                "batch": pd.Categorical(
                    ["B"] * 4 + ["A"] * 4, categories=["B", "A", "unused"]
                ),
                "treatment": ["control", "control", "treated", "treated"] * 2,
                "subject": [1, 2, 1, 2] * 2,
            },
            index=pd.Index(range(8, 0, -1), name="sample"),
        ),
    )


def assert_summary(ax, x, spectra, method, *, index=0, ignore_na=False):
    if method == "plot_mean_sd":
        mean = np.nanmean if ignore_na else np.mean
        std = np.nanstd if ignore_na else np.std
        center = mean(spectra, axis=0)
        lower, upper = center - std(spectra, axis=0), center + std(spectra, axis=0)
    else:
        quantile = np.nanquantile if ignore_na else np.quantile
        lower, center, upper = quantile(spectra, [0.25, 0.5, 0.75], axis=0)

    np.testing.assert_array_equal(ax.lines[index].get_xdata(), x)
    np.testing.assert_allclose(ax.lines[index].get_ydata(), center)
    paths = ax.collections[index].get_paths()
    vertices = np.concatenate([path.vertices for path in paths])
    for coordinate, low, high in zip(x, lower, upper):
        if np.isnan(low) or np.isnan(high):
            continue
        bounds = vertices[vertices[:, 0] == coordinate, 1]
        np.testing.assert_allclose([bounds.min(), bounds.max()], [low, high])


def test_aggregate_plot_summarizes_all_spectra_without_changing_frame(frame, method):
    original_spc, original_data = frame.spc.copy(), frame.data.copy(deep=True)

    fig, ax = getattr(frame, method)(color="black", fill_kws={"alpha": 0.15})

    assert fig.axes == [ax]
    assert len(ax.lines) == len(ax.collections) == 1
    assert_summary(ax, frame.wl, frame.spc, method)
    np.testing.assert_allclose(
        ax.collections[0].get_facecolor(), [to_rgba("black", 0.15)]
    )
    np.testing.assert_array_equal(frame.spc, original_spc)
    pd.testing.assert_frame_equal(frame.data, original_data)


@pytest.mark.parametrize("col", [None, "batch"])
def test_custom_aggregate_with_ribbon_supports_defaults_facets_and_line_styles(
    frame, col
):
    def aggregate(spectra):
        return np.stack(
            (spectra.min(axis=0), spectra.mean(axis=0), spectra.max(axis=0))
        )

    _, axes = frame.plot_aggregate_with_ribbon(
        aggregate, col=col, linewidth=2.5, linestyle="--"
    )

    groups = [range(8)] if col is None else [range(4), range(4, 8)]
    for ax, indices in zip(np.atleast_1d(axes).flat, groups):
        lower, center, upper = aggregate(frame.spc[indices])
        assert len(ax.lines) == len(ax.collections) == 1
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), frame.wl)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), center)
        assert ax.lines[0].get_linewidth() == 2.5
        assert ax.lines[0].get_linestyle() == "--"
        vertices = ax.collections[0].get_paths()[0].vertices
        for x, low, high in zip(frame.wl, lower, upper):
            bounds = vertices[vertices[:, 0] == x, 1]
            np.testing.assert_allclose([bounds.min(), bounds.max()], [low, high])


@pytest.mark.parametrize("groupby", [None, "subject", ["subject", "batch", "subject"]])
def test_aggregate_plot_groups_by_roles_plus_explicit_columns(frame, method, groupby):
    fig, axes = getattr(frame, method)(
        row="batch", col="treatment", hue="treatment", groupby=groupby
    )

    assert axes.shape == (2, 2)
    assert [ax.get_title() for ax in axes.flat] == [
        "batch = B | treatment = control",
        "batch = B | treatment = treated",
        "batch = A | treatment = control",
        "batch = A | treatment = treated",
    ]
    assert len(fig.legends) == 1
    for ax, indices in zip(axes.flat, [[0, 1], [2, 3], [4, 5], [6, 7]]):
        groups = [indices] if groupby is None else [[i] for i in indices]
        assert len(ax.lines) == len(ax.collections) == len(groups)
        for index, group in enumerate(groups):
            assert_summary(ax, frame.wl, frame.spc[group], method, index=index)


def test_aggregate_plot_retains_missing_groups_and_literal_labels(frame, method):
    frame.data["batch"] = pd.Categorical(
        ["Missing", None, "Missing", None, "NA", "NA", pd.NA, np.nan],
        categories=["NA", "Missing", "unused"],
    )

    _, axes = getattr(frame, method)(col="batch", hue="batch", col_wrap=2)

    assert axes.shape == (2, 2)
    for ax, title, indices in zip(
        axes.flat,
        ["batch = NA", "batch = Missing", "batch = Missing"],
        [[4, 5], [0, 2], [1, 3, 6, 7]],
    ):
        assert ax.get_title() == title
        assert len(ax.lines) == len(ax.collections) == 1
        assert_summary(ax, frame.wl, frame.spc[indices], method)
    assert to_rgba(axes[1, 0].lines[0].get_color()) == to_rgba("gray")
    assert not axes[1, 1].get_visible()


def test_aggregate_plot_preserves_external_vectors_and_continuous_hue(frame, method):
    membership = pd.Series(
        pd.Categorical(["X", "X", "Y", "Y"] * 2, categories=["Y", "X"]),
        index=range(20, 28),
        name="External",
    )
    hue = pd.Series([10] * 4 + [20] * 4, dtype="Int64", name="Dose")

    fig, axes = getattr(frame, method)(col=membership, hue=hue)

    assert [ax.get_title() for ax in axes.flat] == ["External = Y", "External = X"]
    for ax, groups in zip(axes.flat, [[[2, 3], [6, 7]], [[0, 1], [4, 5]]]):
        assert len(ax.lines) == len(ax.collections) == 2
        for index, indices in enumerate(groups):
            assert_summary(ax, frame.wl, frame.spc[indices], method, index=index)
    assert fig.axes[-1].get_ylabel() == "Dose"
    np.testing.assert_allclose(fig.axes[-1].get_ylim(), [10, 20])


def test_aggregate_plot_groups_without_assigning_plot_roles(frame, method):
    _, ax = getattr(frame, method)(groupby=["batch", "treatment"], legend=False)

    assert len(ax.lines) == len(ax.collections) == 4
    assert ax.get_legend() is None
    for index, group in enumerate([[0, 1], [2, 3], [4, 5], [6, 7]]):
        assert_summary(ax, frame.wl, frame.spc[group], method, index=index)


@pytest.mark.parametrize("ignore_na", [False, True])
def test_aggregate_plot_controls_missing_intensities(frame, method, ignore_na):
    frame.spc[0, 0] = np.nan

    _, ax = getattr(frame, method)(ignore_na=ignore_na)

    assert_summary(ax, frame.wl, frame.spc, method, ignore_na=ignore_na)


@pytest.mark.parametrize("ddof", [0, 1])
def test_mean_sd_accepts_ddof(frame, ddof):
    _, ax = frame.plot_mean_sd(ddof=ddof)

    vertices = ax.collections[0].get_paths()[0].vertices
    lower = frame.spc.mean(axis=0) - frame.spc.std(axis=0, ddof=ddof)
    upper = frame.spc.mean(axis=0) + frame.spc.std(axis=0, ddof=ddof)
    for x, low, high in zip(frame.wl, lower, upper):
        bounds = vertices[vertices[:, 0] == x, 1]
        np.testing.assert_allclose([bounds.min(), bounds.max()], [low, high])


def test_quantiles_accepts_custom_lower_center_and_upper_probabilities(frame):
    _, ax = frame.plot_quantiles(q=(0, 0.4, 1))

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), np.quantile(frame.spc, 0.4, axis=0)
    )
    vertices = ax.collections[0].get_paths()[0].vertices
    for x, low, high in zip(frame.wl, frame.spc.min(axis=0), frame.spc.max(axis=0)):
        bounds = vertices[vertices[:, 0] == x, 1]
        np.testing.assert_allclose([bounds.min(), bounds.max()], [low, high])


@pytest.mark.parametrize(
    "q", [0.5, (0.25, 0.75), (0.5, 0.25, 1), (-0.1, 0.5, 1), (0, 0.5, np.nan)]
)
def test_quantiles_rejects_invalid_probabilities_before_plotting(frame, q):
    figures = plt.get_fignums()

    with pytest.raises(ValueError, match="q must"):
        frame.plot_quantiles(q=q)

    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "options, error",
    [
        ({"groupby": "unknown"}, ValueError),
        ({"row": "unknown"}, KeyError),
        ({"hue": [1, 2]}, ValueError),
        ({"ymin": [[0, 0, 0]]}, ValueError),
    ],
)
def test_aggregate_plot_rejects_invalid_arguments_before_plotting(
    frame, method, options, error
):
    figures = plt.get_fignums()

    with pytest.raises(error):
        getattr(frame, method)(**options)

    assert plt.get_fignums() == figures


def test_aggregate_plot_rejects_empty_frame(method):
    frame = SpectraFrame(np.empty((0, 3)))

    with pytest.raises(ValueError, match="empty"):
        getattr(frame, method)()
