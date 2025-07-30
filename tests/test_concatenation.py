import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
import pytest

from pyspc import SpectraFrame, concat


def test_concat_axis0():
    sf1 = SpectraFrame(
        np.array([[1, 2, 3], [4, 5, 6]]),
        [100, 200, 300],
        pd.DataFrame({"A": [7, 8], "B": [9, 10]}),
    )
    sf2 = SpectraFrame(
        np.array([[7, 8, 9], [10, 11, 12]]),
        [400, 500, 600],
        pd.DataFrame({"A": [9, 10], "B": [11, 12]}),
    )

    # Different wavelenghts
    with pytest.raises(ValueError):
        concat([sf1, sf2], axis=0)

    sf1.wl = np.array([400, 500, 600])

    sf_all = concat([sf1, sf2], axis=0)
    assert_array_equal(sf_all.spc, np.arange(1, 13).reshape((-1, 3)))
    assert_array_equal(sf_all.wl, sf1.wl)
    assert_frame_equal(
        sf_all.data, pd.DataFrame({"A": [7, 8, 9, 10], "B": [9, 10, 11, 12]})
    )


def test_concat_axis1():
    sf1 = SpectraFrame(
        np.array([[1, 2, 3], [7, 8, 9]]),
        [100, 200, 300],
        pd.DataFrame({"A": [7, 8], "B": [9, 10]}, index=[0, 1]),
    )
    sf2 = SpectraFrame(
        np.array([[4, 5, 6], [10, 11, 12]]),
        [400, 500, 600],
        pd.DataFrame({"C": [11, 12], "D": [13, 14]}, index=[3, 4]),
    )

    sf_all = concat([sf1, sf2], axis=1)
    assert_array_equal(sf_all.spc, np.arange(1, 13).reshape((-1, 6)))
    assert_array_equal(sf_all.wl, np.arange(100, 601, 100))
    assert_frame_equal(
        sf_all.data,
        pd.concat(
            [sf1.data.reset_index(drop=True), sf2.data.reset_index(drop=True)],
            axis=1,
            ignore_index=False,
        ),
    )


def test_concat_axis1_same_data():
    sf = SpectraFrame(
        np.array([[1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18]]),
        [100, 200, 300, 400, 500, 600, 700, 800],
        pd.DataFrame({"A": [7, 8], "B": [9, 10]}),
    )

    # Overlapping data columns
    with pytest.raises(ValueError):
        concat([sf, sf], axis=1)

    sf_cutted = concat(
        [
            sf[:, :, :300],
            sf[:, [], 700:],
        ],
        axis=1,
    )
    assert_array_equal(sf_cutted.spc, sf.spc[:, [0, 1, 2, 6, 7]])
    assert_array_equal(sf_cutted.wl, [100, 200, 300, 700, 800])
    assert_frame_equal(sf_cutted.data, sf.data)
