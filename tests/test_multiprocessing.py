import numpy as np
from pandas.testing import assert_frame_equal
import pytest

from pyspc import SpectraFrame


def _row_minmax(x: np.ndarray) -> np.ndarray:
    return np.array([np.min(x), np.max(x)])


def _row_sum(x: np.ndarray) -> float:
    return float(np.sum(x))


def test_baseline_parallel_matches_serial():
    wl = 400 + np.arange(0, 10, 2)
    signal = np.array([0, 5, 10, 5, 0])
    bl = 10 + 5 * (wl - 400)
    spc = np.tile(signal + bl, (100, 1))
    sf = SpectraFrame(spc, wl=wl)
    expected = sf.baseline("rubberband")

    try:
        result = sf.baseline(
            "rubberband",
            parallel=True,
            max_workers=2,
        )
    except (PermissionError, OSError) as e:
        pytest.skip(f"ProcessPoolExecutor is not available: {e}")
    assert np.array_equal(result.spc, expected.spc)
    assert np.array_equal(result.spc[0, :], bl)


def test_apply_parallel_matches_serial_for_callable():
    sf = SpectraFrame(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        wl=np.array([400.0, 500.0, 600.0]),
        data={"group": ["A", "A", "B"]},
    )
    expected = sf.apply(_row_minmax, axis=1)

    try:
        result = sf.apply(_row_minmax, axis=1, parallel=True, max_workers=2)
    except (PermissionError, OSError) as e:
        pytest.skip(f"ProcessPoolExecutor is not available: {e}")

    assert np.array_equal(result.spc, expected.spc)
    assert np.array_equal(result.wl, expected.wl)
    assert_frame_equal(result.data, expected.data)


def test_apply_parallel_rejects_numpy_vectorized_function():
    sf = SpectraFrame(np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError, match="non-vectorized"):
        sf.apply("mean", axis=1, parallel=True)


def test_apply_parallel_rejects_axis_0():
    sf = SpectraFrame(np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError, match="axis=1"):
        sf.apply(_row_sum, axis=0, parallel=True)


def test_apply_parallel_rejects_groupby():
    sf = SpectraFrame(np.arange(12).reshape(3, 4), data={"group": ["A", "A", "B"]})
    with pytest.raises(ValueError, match="groupby"):
        sf.apply(_row_sum, groupby="group", parallel=True)


def test_apply_parallel_rejects_non_pickleable_callable():
    sf = SpectraFrame(np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError, match="pickleable"):
        sf.apply(lambda x: float(np.sum(x)), axis=1, parallel=True)


def test_apply_parallel_rejects_invalid_chunksize():
    sf = SpectraFrame(np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError, match="chunksize"):
        sf.apply(_row_sum, axis=1, parallel=True, chunksize=0)
