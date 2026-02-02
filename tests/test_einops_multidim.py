import numpy as np
import pandas as pd
import pytest

from pyspc import SpectraFrame


def _make_cube_sf(batch: int = 2, y: int = 2, x: int = 3, nwl: int = 4) -> SpectraFrame:
    wl = np.linspace(600, 700, nwl)
    rows = []
    spc_rows = []
    for b in range(batch):
        for yy in range(y):
            for xx in range(x):
                base = 100 * b + 10 * yy + xx
                spc_rows.append(base + np.arange(nwl, dtype=float))
                rows.append({"batch": b, "y": yy, "x": xx})
    return SpectraFrame(np.asarray(spc_rows), wl=wl, data=pd.DataFrame(rows))


@pytest.fixture()
def einops_module():
    """Provide optional einops dependency for tests that require it."""
    return pytest.importorskip("einops")


@pytest.fixture()
def cube_sf():
    """Provide a basic cube SpectraFrame for tests."""
    return _make_cube_sf()


def test_fill_missing_grid_validates_columns():
    sf = SpectraFrame(
        spc=[[1, 2], [3, 4]], wl=[400, 500], data={"x": [0, 1], "y": [0, 1]}
    )

    for val in [None, 1, "", [], ["z"], ["x", "z"]]:
        with pytest.raises(ValueError):
            sf._fill_missing_grid(columns=val)

    for grid_val in [{"z": [0, 1]}, {"x": [0, 1], "z": [0, 1]}]:
        with pytest.raises(ValueError):
            sf._fill_missing_grid(columns=["x", "y"], grid_values=grid_val)

    for grid_val in [{"x": None}, {"x": 1}, {"x": ""}, {"x": [0, 0]}]:
        with pytest.raises(ValueError):
            sf._fill_missing_grid(columns=["x", "y"], grid_values=grid_val)


def test_fill_missing_grid_contains_at_least_one_spectrum(cube_sf):
    # Filling an empty SpectraFrame should raise an error.
    sf = SpectraFrame(
        spc=np.empty((0, 2)), wl=[400, 500], data=pd.DataFrame(columns=["x", "y"])
    )

    with pytest.raises(ValueError):
        sf._fill_missing_grid(columns=["x", "y"])

    # non-overlapping grid values should raise an error
    with pytest.raises(ValueError):
        cube_sf._fill_missing_grid(
            columns=["x", "y"], grid_values={"x": [200, 300], "y": [200, 300]}
        )


def test_fill_missing_grid_adds_missing_combinations():
    # Fill a ragged (x, y) grid by inserting a missing coordinate combination.
    sf = SpectraFrame(
        spc=[[1, 2], [3, 4], [5, 6]],
        wl=[400, 500],
        data={"x": [0, 0, 1], "y": [0, 1, 0]},
    )

    filled = sf._fill_missing_grid(columns=["x", "y"])
    assert filled.nspc == 4

    expected_xy = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
    pd.testing.assert_frame_equal(
        filled.data.loc[:, ["x", "y"]],
        expected_xy,
    )
    assert np.isnan(filled.spc[-1]).all()


def test_fill_missing_grid_drops_non_axis_metadata_columns():
    sf = SpectraFrame(
        spc=[[1, 2], [3, 4], [5, 6]],
        wl=[400, 500],
        data={"x": [0, 0, 1], "y": [0, 1, 0], "label": ["a", "b", "c"]},
    )

    filled = sf._fill_missing_grid(columns=["x", "y"])
    assert filled.data.columns.tolist() == ["x", "y"]


def test_prepare_for_einops_validates_pattern(cube_sf):
    # Invalid patterns should raise errors.
    for pattern in [
        None,
        "",
        "invalid pattern!",
        "x y",
        "x y z wl",
        "x x wl",
        "(x y batch) wl -> x y batch wl",
    ]:
        with pytest.raises(ValueError):
            cube_sf._prepare_for_einops(reduction="rearrange", pattern=pattern)


def test_prepare_for_einops_builds_pattern_and_sorts():
    # Prepare sorted spectra and sizes for a basic (y, x, wl) rearrangement.
    sf = SpectraFrame(
        spc=np.arange(4 * 5).reshape((4, 5)),
        wl=[400, 500, 600, 700, 800],
        data={"y": [1, 0, 1, 0], "x": [0, 0, 1, 1]},
    )

    sorted_spc, einops_pattern, sizes = sf._prepare_for_einops(
        reduction="rearrange",
        pattern="y x wl",
    )

    assert einops_pattern == "(y x) wl -> y x wl"
    assert sizes == {"y": 2, "x": 2, "wl": 5}

    expected_sorted = np.array(
        [
            [5, 6, 7, 8, 9],
            [15, 16, 17, 18, 19],
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
        ],
    )
    np.testing.assert_allclose(sorted_spc, expected_sorted)


def test_prepare_for_einops_rejects_ellipsis():
    # Ellipsis is intentionally unsupported for now.
    sf = _make_cube_sf()
    with pytest.raises(NotImplementedError):
        sf._prepare_for_einops(reduction="rearrange", pattern="... wl")


def test_prepare_for_einops_detects_incomplete_dimensions(cube_sf):
    # Missing grid dimensions should raise an error.
    for pattern in [
        "batch wl",
        "(batch y) wl",
        "(y x) wl",
    ]:
        with pytest.raises(ValueError):
            cube_sf._prepare_for_einops(reduction="rearrange", pattern=pattern)


def test_get_einops_rest_column_avoids_name_collision():
    # Ensure `_get_einops_rest_column` does not collide with existing columns.
    sf = SpectraFrame(
        spc=np.arange(2 * 2).reshape((2, 2)),
        wl=[400, 500],
        data=pd.DataFrame({"y": [0, 0], "_einops_rest": ["a", "b"]}),
    )

    rest = sf._get_einops_rest_column(kept_columns=["y"])
    assert rest.name != "_einops_rest"
    assert isinstance(rest.dtype, pd.CategoricalDtype)


def test_rearrange_matches_explicit_einops(einops_module):
    einops = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    out = sf.rearrange("(batch y) x wl")
    ref = einops.rearrange(
        sf.sort_values(["batch", "y", "x"]).spc,
        "(batch y x) wl -> (batch y) x wl",
        batch=2,
        y=2,
        x=3,
    )
    np.testing.assert_allclose(out, ref)


def test_rearrange_is_deterministic_under_row_shuffle(einops_module):
    _ = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    shuffled = sf.assign(_rand=np.random.RandomState(0).rand(sf.nspc)).sort_values(
        "_rand"
    )
    shuffled = shuffled.drop(columns=["_rand"])
    a = sf.rearrange("batch y x wl")
    b = shuffled.rearrange("batch y x wl")
    np.testing.assert_allclose(a, b)


def test_rearrange_requires_wl(einops_module):
    _ = einops_module
    sf = _make_cube_sf()
    with pytest.raises(ValueError):
        sf.rearrange("batch y x")


def test_rearrange_duplicate_coordinates_error(einops_module):
    _ = einops_module
    sf = _make_cube_sf()
    sf_dup = SpectraFrame(
        np.concatenate([sf.spc, sf.spc], axis=0),
        wl=sf.wl,
        data=pd.concat([sf.data, sf.data], axis=0, ignore_index=True),
    )
    with pytest.raises(ValueError, match="Duplicate coordinate combinations"):
        sf_dup.rearrange("batch y x wl")


def test_rearrange_ragged_grid_fills_with_nan_by_default(einops_module):
    _ = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    # Drop one coordinate tuple: (batch=0, y=0, x=0).
    sf_ragged = sf.query("not (batch == 0 and y == 0 and x == 0)")
    out = sf_ragged.rearrange("(batch y) x wl")
    assert out.shape == (4, 3, sf.nwl)
    assert np.isnan(out[0, 0, :]).all()


def test_rearrange_fill_value_for_ragged_grid(einops_module):
    _ = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    # Drop one coordinate tuple: (batch=0, y=0, x=0)
    sf_ragged = sf.query("not (batch == 0 and y == 0 and x == 0)")
    out = sf_ragged.rearrange("(batch y) x wl", fill_value=0.0)
    assert out.shape == (4, 3, sf.nwl)
    assert np.all(out[0, 0, :] == 0.0)


def test_rearrange_allows_singleton_axis(einops_module):
    # Allow adding constant (singleton) axes via einops constants in the pattern.
    _ = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    out = sf.rearrange("batch y x 1 wl")
    assert out.shape == (2, 2, 3, 1, sf.nwl)

    ref = sf.rearrange("batch y x wl")
    np.testing.assert_allclose(out[..., 0, :], ref)


def test_rearrange_grid_values_can_expand_grid(einops_module):
    # Explicit grid values can expand the requested tensor shape.
    _ = einops_module
    sf = _make_cube_sf(batch=1, y=2, x=3, nwl=4)
    out = sf.rearrange("y x wl", fill_value=np.nan, x=[0, 1, 2, 3])
    assert out.shape == (2, 4, sf.nwl)
    assert np.isnan(out[:, 3, :]).all()


def test_rearrange_grid_values_sorted_accordingly(einops_module):
    # Explicit grid values can expand the requested tensor shape.
    _ = einops_module
    sf = _make_cube_sf(batch=1, y=2, x=2, nwl=4)
    out = sf.rearrange("y x wl", fill_value=np.nan, x=[3, 1, 2, 0])
    unexpanded = sf.rearrange("y x wl")
    assert out.shape == (2, 4, sf.nwl)
    assert np.isnan(out[:, 0, :]).all()
    np.testing.assert_allclose(out[:, 1, :], unexpanded[:, 1, :])
    assert np.isnan(out[:, 2, :]).all()
    np.testing.assert_allclose(out[:, 3, :], unexpanded[:, 0, :])


def test_reduce_mean_matches_einops_reduce_over_wl(einops_module):
    einops = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    out = sf.reduce("mean", "(batch y) x")
    cube = sf.rearrange("batch y x wl")
    ref = einops.reduce(cube, "batch y x wl -> (batch y) x", "mean")
    np.testing.assert_allclose(out, ref)


def test_reduce_accepts_callable_reducer(einops_module):
    # Callable reducers should behave like string reducers for NumPy equivalents.
    _ = einops_module
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    out_callable = sf.reduce(np.mean, "(batch y) x")
    out_str = sf.reduce("mean", "(batch y) x")
    np.testing.assert_allclose(out_callable, out_str)


def test_reduce_validates_reducer():
    sf = _make_cube_sf(batch=2, y=2, x=3, nwl=4)
    for reducer in [None, 123, 3.14, [], {}, "unknown_reducer", ""]:
        with pytest.raises(ValueError):
            sf.reduce(reducer, "(batch y) x")


def test_reduce_uses_remaining_columns_as_rest_axis_with_fill_value(einops_module):
    _ = einops_module
    wl = np.linspace(600, 700, 4)
    rows = []
    spc_rows = []
    for yy in range(2):
        for xx in range(2):
            for batch in range(2):
                for rep in range(2):
                    # Drop one combination to create a ragged "rest" grid.
                    if (yy, xx, batch, rep) == (0, 0, 1, 1):
                        continue

                    base = 100 * yy + 10 * xx + 2 * batch + rep
                    spc_rows.append(base + np.arange(len(wl), dtype=float))
                    rows.append({"y": yy, "x": xx, "batch": batch, "rep": rep})
    sf = SpectraFrame(np.asarray(spc_rows), wl=wl, data=pd.DataFrame(rows))

    out = sf.reduce("mean", "y x wl", ignore_na=True)
    assert out.shape == (2, 2, sf.nwl)

    # Compute reference mean spectrum per (y, x) ignoring missing combinations.
    ref = np.full((2, 2, sf.nwl), np.nan, dtype=float)
    for yy in range(2):
        for xx in range(2):
            mask = (sf.data["y"] == yy) & (sf.data["x"] == xx)
            ref[yy, xx, :] = sf.spc[mask, :].mean(axis=0)
    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_reduce_ignore_na_false_propagates_nan(einops_module):
    # Without NaN-aware reduction, any missing rest entries propagate NaNs.
    _ = einops_module
    wl = np.linspace(600, 700, 4)
    rows = []
    spc_rows = []
    for yy in range(2):
        for xx in range(2):
            for batch in range(2):
                for rep in range(2):
                    if (yy, xx, batch, rep) == (0, 0, 1, 1):
                        continue
                    base = 100 * yy + 10 * xx + 2 * batch + rep
                    spc_rows.append(base + np.arange(len(wl), dtype=float))
                    rows.append({"y": yy, "x": xx, "batch": batch, "rep": rep})
    sf = SpectraFrame(np.asarray(spc_rows), wl=wl, data=pd.DataFrame(rows))

    out = sf.reduce("mean", "y x wl", fill_value=np.nan, ignore_na=False)
    assert np.isnan(out[0, 0, :]).all()
    assert not np.isnan(out[0, 1, :]).any()


def test_reduce_ragged_grid_fills_with_nan_by_default(einops_module):
    # Ragged grids are padded with NaNs by default.
    _ = einops_module
    wl = np.linspace(600, 700, 4)
    rows = []
    spc_rows = []
    for yy in range(2):
        for xx in range(2):
            for batch in range(2):
                for rep in range(2):
                    if (yy, xx, batch, rep) == (0, 0, 1, 1):
                        continue
                    base = 100 * yy + 10 * xx + 2 * batch + rep
                    spc_rows.append(base + np.arange(len(wl), dtype=float))
                    rows.append({"y": yy, "x": xx, "batch": batch, "rep": rep})
    sf = SpectraFrame(np.asarray(spc_rows), wl=wl, data=pd.DataFrame(rows))

    out = sf.reduce("mean", "y x wl", ignore_na=True)
    assert out.shape == (2, 2, sf.nwl)


def test_reduce_unknown_reducer_string(einops_module):
    _ = einops_module
    sf = _make_cube_sf()
    with pytest.raises(ValueError):
        sf.reduce("does_not_exist", "(batch y) x")
