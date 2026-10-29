"""Facet title formatters receive category values as two positional arguments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyspc.plotting import (
    _default_title_formatter,
    _generate_facet_titles,
    plot_spectra,
)


@pytest.fixture(autouse=True)
def close_new_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


@pytest.mark.parametrize(
    "row_value, col_value, names, expected",
    [
        ("A", 2, {}, "A | 2"),
        ("A", 2, {"row_name": "Group", "col_name": "Time"}, "Group = A | Time = 2"),
        (None, 2, {"col_name": "Time"}, "Time = 2"),
        (pd.NA, None, {"row_name": "Group"}, "Group = Missing"),
        (None, None, {}, ""),
        ("", 2, {}, "2"),
    ],
)
def test_default_title_formatter(row_value, col_value, names, expected):
    assert _default_title_formatter(row_value, col_value, **names) == expected


@pytest.mark.parametrize(
    "formatter, expected",
    [
        (_default_title_formatter, [["2 | B", "2 | A"], ["1 | B", "1 | A"]]),
        (
            lambda row_value, col_value: f"{row_value}:{col_value}",
            [["2:B", "2:A"], ["1:B", "1:A"]],
        ),
    ],
)
def test_generate_facet_titles_preserves_row_and_column_order(formatter, expected):
    assert _generate_facet_titles([2, 1], ["B", "A"], formatter) == expected


def test_title_formatter_receives_original_values_for_every_rectangular_panel():
    earlier, later = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")
    row = pd.Series([2, 1, 2], index=[3, 2, 1], name="Unused row name")
    col = pd.Series(
        pd.Categorical([earlier, later, earlier], categories=[later, earlier]),
        name="Unused column name",
    )
    calls = []

    def formatter(row_value, col_value, /):
        assert isinstance(row_value, (int, np.integer))
        assert isinstance(col_value, pd.Timestamp)
        calls.append((row_value, col_value))
        return f"Group {row_value}: {col_value:%Y-%m}"

    _, axes = plot_spectra(
        np.arange(6).reshape(3, 2), row=row, col=col, title_formatter=formatter
    )

    assert calls == [(1, later), (1, earlier), (2, later), (2, earlier)]
    assert [ax.get_title() for ax in axes.flat] == [
        "Group 1: 2025-02",
        "Group 1: 2025-01",
        "Group 2: 2025-02",
        "Group 2: 2025-01",
    ]
    assert [len(ax.lines) for ax in axes.flat] == [1, 0, 0, 2]


@pytest.mark.parametrize("dimension", ["row", "col"])
def test_title_formatter_receives_none_for_the_unused_dimension(dimension):
    calls = []

    def formatter(row_value, col_value):
        calls.append((row_value, col_value))
        return f"{row_value!r}/{col_value!r}"

    _, axes = plot_spectra(
        [[1, 2], [3, 4], [5, 6]], **{dimension: [2, 1, 2]}, title_formatter=formatter
    )

    expected = [(1, None), (2, None)] if dimension == "row" else [(None, 1), (None, 2)]
    assert calls == expected
    assert [ax.get_title() for ax in axes.flat] == [
        f"{row_value!r}/{col_value!r}" for row_value, col_value in expected
    ]


def test_wrapped_title_formatter_skips_hidden_panels_and_unused_categories():
    col = pd.Categorical([5, 3, 1, 4, 2], categories=[1, 2, 3, 4, 5, 6])
    calls = []

    def formatter(row_value, col_value):
        calls.append((row_value, col_value))
        return f"Group {col_value:02d}"

    _, axes = plot_spectra(
        np.arange(10).reshape(5, 2), col=col, col_wrap=2, title_formatter=formatter
    )

    assert calls == [(None, value) for value in range(1, 6)]
    assert [ax.get_title() for ax in axes.flat] == [
        "Group 01",
        "Group 02",
        "Group 03",
        "Group 04",
        "Group 05",
        "",
    ]
    assert not axes[2, 1].get_visible()


def test_title_formatter_can_return_empty_titles():
    _, axes = plot_spectra(
        [[1, 2], [3, 4]],
        col=["A", "B"],
        title_formatter=lambda row_value, col_value: "",
    )

    assert all(ax.get_title() == "" for ax in axes.flat)
    assert all(len(ax.lines) == 1 for ax in axes.flat)


@pytest.mark.parametrize("formatter", [False, 1, "title", {}])
def test_title_formatter_must_be_callable_before_creating_a_figure(formatter):
    figures = plt.get_fignums()

    with pytest.raises(TypeError, match="title_formatter.*callable"):
        plot_spectra([[1, 2]], col=["A"], title_formatter=formatter)

    assert plt.get_fignums() == figures


@pytest.mark.parametrize("result", [None, 1, ["title"], {"label": "title"}])
def test_title_formatter_must_return_a_string_before_creating_a_figure(result):
    figures = plt.get_fignums()

    def formatter(row_value, col_value):
        return result

    with pytest.raises(TypeError, match="title_formatter.*str"):
        plot_spectra([[1, 2]], col=["A"], title_formatter=formatter)

    assert plt.get_fignums() == figures


def test_title_formatter_exception_is_propagated_without_creating_a_figure():
    figures = plt.get_fignums()
    error = RuntimeError("Cannot format this category")

    def formatter(row_value, col_value):
        raise error

    with pytest.raises(RuntimeError) as caught:
        plot_spectra([[1, 2]], col=["A"], title_formatter=formatter)

    assert caught.value is error
    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "dimension, col_wrap", [("row", None), ("col", None), ("col", 1)]
)
def test_title_formatter_distinguishes_missing_membership_from_literal_missing(
    dimension, col_wrap
):
    calls = []

    def formatter(row_value, col_value):
        value = row_value if dimension == "row" else col_value
        unused = col_value if dimension == "row" else row_value
        assert unused is None
        calls.append(value)
        return "Missing membership" if value is pd.NA else f"Literal {value}"

    _, axes = plot_spectra(
        np.arange(8).reshape(4, 2),
        **{dimension: ["Missing", None, np.nan, pd.NA]},
        col_wrap=col_wrap,
        title_formatter=formatter,
    )

    assert calls[0] == "Missing"
    assert calls[1] is pd.NA
    assert len(calls) == 2
    assert [ax.get_title() for ax in axes.flat] == [
        "Literal Missing",
        "Missing membership",
    ]
    assert [len(ax.lines) for ax in axes.flat] == [1, 3]


def test_title_formatter_receives_pd_na_when_both_facet_values_are_missing():
    calls = []

    def formatter(row_value, col_value):
        assert row_value is pd.NA
        assert col_value is pd.NA
        calls.append((row_value, col_value))
        return "No metadata"

    _, axes = plot_spectra(
        [[1, 2]], row=[None], col=[np.nan], title_formatter=formatter
    )

    assert len(calls) == 1
    assert axes[0, 0].get_title() == "No metadata"


@pytest.mark.parametrize("supplied_axis", [False, True])
def test_title_formatter_is_not_called_without_facets(supplied_axis):
    ax = None
    if supplied_axis:
        _, ax = plt.subplots()
        ax.set_title("Existing title")

    def formatter(row_value, col_value):
        pytest.fail("The formatter should only run for facet grids")

    _, returned_ax = plot_spectra([[1, 2]], ax=ax, title_formatter=formatter)

    assert returned_ax.get_title() == ("Existing title" if supplied_axis else "")
