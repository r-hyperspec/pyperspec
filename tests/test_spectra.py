import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
import pytest
from pathlib import Path
from pyspc import SpectraFrame
from pyspc.testing import assert_spectraframe_equal


class TestSpectraFrameInit:
    @pytest.mark.parametrize(
        "spc",
        [range(1, 6), list(range(1, 6)), np.arange(1, 6), pd.Series(range(1, 6))],
    )
    def test_spc_1d(self, spc):
        sf = SpectraFrame(spc)
        assert isinstance(sf.spc, np.ndarray)
        assert sf.spc.shape == (1, 5)
        assert_array_equal(sf.spc[0, :], np.arange(1, 6))
        assert_array_equal(sf.wl, np.arange(5))
        assert_frame_equal(sf.data, pd.DataFrame(index=[0], columns=None))

    def test_spc_2d(self):
        spc = [[1, 2, 3], [4, 5, 6]]
        sf = SpectraFrame(spc)
        assert isinstance(sf.spc, np.ndarray)
        assert_array_equal(sf.spc, np.array(spc))
        assert_array_equal(sf.wl, np.arange(3))
        assert_frame_equal(sf.data, pd.DataFrame(index=[0, 1], columns=None))

    @pytest.mark.parametrize(
        "wl",
        [range(1, 6), list(range(1, 6)), np.arange(1, 6), pd.Series(range(1, 6))],
    )
    def test_wl(self, wl):
        sf = SpectraFrame(range(5), wl=wl)
        assert_array_equal(sf.spc[0, :], np.arange(5))
        assert_array_equal(sf.wl, np.arange(1, 6))
        assert_frame_equal(sf.data, pd.DataFrame(index=[0], columns=None))

    def test_data(self):
        spc = [[1, 2, 3], [4, 5, 6]]
        sf = SpectraFrame(spc, data=pd.Series(["A", "B"], name="group"))
        assert_frame_equal(
            sf.data, pd.DataFrame(["A", "B"], index=[0, 1], columns=["group"])
        )

        sf = SpectraFrame(
            spc, wl=[10, 20, 30], data=pd.Series(["A", "B"], name="group")
        )
        assert_frame_equal(
            sf.data, pd.DataFrame(["A", "B"], index=[0, 1], columns=["group"])
        )

    @pytest.mark.parametrize(
        "spc,wl,data",
        [
            ([1, 2, 3], [0], None),
            ([1, 2, 3], [0, 1, 2, 3, 4], None),
            ([1, 2, 3], None, pd.Series(["A", "B"], name="group")),
        ],
    )
    def test_invalid(self, spc, wl, data):
        with pytest.raises(ValueError):
            SpectraFrame(spc, wl, data)


class TestSpectraFrameMath:
    def sf(self) -> SpectraFrame:
        spc = np.array([[1, 2, 3], [4, 5, 6]])
        wl = [10, 20, 30]
        data = pd.Series(["A", "B"], name="group")
        return SpectraFrame(spc, wl=wl, data=data)

    def sf2(self) -> SpectraFrame:
        spc2 = 10 * np.array([[1, 2, 3], [4, 5, 6]])
        wl2 = [1, 2, 3]
        data2 = pd.Series(["A1", "A2"], name="attr")
        return SpectraFrame(spc2, wl=wl2, data=data2)

    @pytest.mark.parametrize(
        "op",
        [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
        ],
        ids=["+", "-", "*", "/"],
    )
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_left_right(self, op, side):
        def assert_the_rest_did_not_change():
            assert_array_equal(sf.spc, spc)
            assert_array_equal(sf.wl, wl)
            assert_frame_equal(sf.data, pd.DataFrame(data))
            assert_array_equal(sf_res.wl, sf.wl)
            assert_frame_equal(sf_res.data, sf.data)

        sf, sf2 = self.sf(), self.sf2()
        spc = sf.spc.copy()
        wl = sf.wl.copy()
        data = sf.data.copy()
        spc2 = sf2.spc.copy()

        if side == "left":
            sf_res = op(sf, 10)
            assert_the_rest_did_not_change()
            assert_array_equal(sf_res.spc, op(spc, 10))

            sf_res = op(sf, sf2)
            assert_the_rest_did_not_change()
            assert_array_equal(sf_res.spc, op(spc, spc2))
        elif side == "right":
            sf_res = op(10, sf)
            assert_the_rest_did_not_change()
            assert_array_equal(sf_res.spc, op(10, spc))

    # @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    # def test_inpace_scalar(self, op):
    #     sf = self.sf()
    #     spc = sf.spc.copy()
    #
    #     if op == "+":
    #         sf += 10
    #         assert_array_equal(sf.spc, spc + 10)
    #     elif op == "-":
    #         sf -= 10
    #         assert_array_equal(sf.spc, spc - 10)
    #     elif op == "*":
    #         sf *= 10
    #         assert_array_equal(sf.spc, spc * 10)
    #     elif op == "/":
    #         sf /= 10
    #         assert_array_equal(sf.spc, spc / 10)

    def test_math(self):
        sf = self.sf()
        spc = sf.spc.copy()

        # Test __abs__
        sf.spc = -1 * spc
        result = abs(sf)
        assert np.array_equal(result.spc, spc)

        # Test __round__
        sf.spc = 0.33 * spc
        result = round(sf, 1)
        assert np.array_equal(result.spc, np.round(sf.spc, 1))

        # Test __floor__
        result = sf.__floor__()
        assert np.array_equal(result.spc, np.floor(sf.spc))

        # Test __ceil__
        result = sf.__ceil__()
        assert np.array_equal(result.spc, np.ceil(sf.spc))

        # Test __trunc__
        result = sf.__trunc__()
        assert np.array_equal(result.spc, np.trunc(sf.spc))

        # Test __array__
        result = np.array(sf)
        assert np.array_equal(result, sf.spc)
        assert result is not sf.spc  # Ensure it's a copy, not the same object


class TestSpectraFrameCopy:
    def sf(self) -> SpectraFrame:
        spc = np.array([[1, 2, 3], [4, 5, 6]])
        wl = [10, 20, 30]
        data = pd.Series(["A", "B"], name="group")
        return SpectraFrame(spc, wl=wl, data=data)

    def test_copy(self):
        sf = self.sf()
        copied = sf.copy()

        # Ensure that the copied SpectraFrame is not the same object as the original
        assert copied is not sf

        # Ensure that the copied SpectraFrame has the same data
        assert np.array_equal(copied.spc, sf.spc)
        assert np.array_equal(copied.wl, sf.wl)
        assert copied.data.equals(sf.data)

        # Ensure that the data in the copied is not the same object references
        assert copied.spc is not sf.spc
        assert copied.wl is not sf.wl
        assert copied.data is not sf.data


class TestSpectraFrameItems:
    def sample_spectra_frame(self) -> SpectraFrame:
        spc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        wl = np.array([400, 500, 600])
        data = pd.DataFrame(
            {"A": [10, 11, 12], "B": [13, 14, 15], "C": [16, 17, 18]},
            index=[5, 6, 7],
        )
        return SpectraFrame(spc, wl, data)

    def test_getitem_loc(self):
        frame = self.sample_spectra_frame()

        # Select specific rows, columns, and wavelengths
        result = frame[6, "A", 500]
        assert np.array_equal(result.spc, np.array([[5.0]]))
        assert np.array_equal(result.wl, np.array([500]))
        assert result.data.equals(frame.data.iloc[[1], [0]])

        # Select specific rows and wavelengths, all columns
        result = frame[5:6, :, :500]
        assert np.array_equal(result.spc, np.array([[1.0, 2.0], [4.0, 5.0]]))
        assert np.array_equal(result.wl, np.array([400, 500]))
        assert result.data.equals(frame.data.iloc[0:2, :])

        # Select specific rows, all columns, and a wavelength range
        result = frame[6:7, :, 400:600]
        assert np.array_equal(result.spc, np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        assert np.array_equal(result.wl, np.array([400, 500, 600]))
        assert result.data.equals(frame.data.iloc[1:3, :])

        # Select specific rows, all columns, and all wavelengths
        result = frame[6:7, :, :]
        assert np.array_equal(result.spc, np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        assert np.array_equal(result.wl, frame.wl)
        assert result.data.equals(frame.data.iloc[1:3, :])

        # Select specific rows, all columns, and specific wavelengths
        result = frame[6:7, :, [400, 600]]
        assert np.array_equal(result.spc, frame.spc[1:3, [0, -1]])
        assert np.array_equal(result.wl, [400, 600])
        assert result.data.equals(frame.data.iloc[1:3, :])

        # Test for invalid wavelengths
        with pytest.raises(ValueError):
            frame[:, :, 540]

        result = frame[:, :, 510:550]
        assert result.spc.shape == (3, 0)
        assert result.data.equals(frame.data)

    def test_getitem_iloc(self):
        frame = self.sample_spectra_frame()

        # Select specific rows, columns, and wavelengths
        result = frame[1, 0, 1, True]
        assert np.array_equal(result.spc, np.array([[5.0]]))
        assert np.array_equal(result.wl, np.array([500]))
        assert result.data.equals(frame.data.iloc[[1], [0]])

        # Select specific rows and wavelengths, all columns
        result = frame[0:2, :, :2, True]
        assert np.array_equal(result.spc, np.array([[1.0, 2.0], [4.0, 5.0]]))
        assert np.array_equal(result.wl, np.array([400, 500]))
        assert result.data.equals(frame.data.iloc[0:2, :])

        # Select specific rows, all columns, and a wavelength range
        result = frame[1:3, :, 0:4, True]
        assert np.array_equal(result.spc, np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        assert np.array_equal(result.wl, np.array([400, 500, 600]))
        assert result.data.equals(frame.data.iloc[1:3, :])

        # Select specific rows, all columns, and all wavelengths
        result = frame[1:3, :, :, True]
        assert np.array_equal(result.spc, np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        assert np.array_equal(result.wl, frame.wl)
        assert result.data.equals(frame.data.iloc[1:3, :])

        # Select specific rows, all columns, and specific wavelengths
        result = frame[1:3, :, [0, -1], True]
        assert np.array_equal(result.spc, frame.spc[1:3, [0, -1]])
        assert np.array_equal(result.wl, [400, 600])
        assert result.data.equals(frame.data.iloc[1:3, :])

    def test_getitem_boolean_vectors(self):
        frame = self.sample_spectra_frame()

        # Test for rows (both loc and iloc)
        assert_spectraframe_equal(
            frame[[False, True, False], :, :],
            frame[1, :, :, True],
        )
        assert_spectraframe_equal(
            frame[[False, True, False], :, :, True],
            frame[1, :, :, True],
        )

        # Test for data columns (both loc and iloc)
        assert_spectraframe_equal(
            frame[:, [False, True, False], :],
            frame[:, 1, :, True],
        )
        assert_spectraframe_equal(
            frame[:, [False, True, False], :, True],
            frame[:, 1, :, True],
        )

        # Test for wl columns (both loc and iloc)
        assert_spectraframe_equal(
            frame[:, :, [False, True, False]],
            frame[:, :, 1, True],
        )
        assert_spectraframe_equal(
            frame[:, :, [False, True, False], True],
            frame[:, :, 1, True],
        )

        # Test for pd.Series
        assert_spectraframe_equal(
            frame[frame.A == 11, :, :],
            frame[1, :, :, True],
        )

        # Test for numpy arrays
        assert_spectraframe_equal(
            frame[np.array(frame.A == 11), :, :],
            frame[1, :, :, True],
        )

    def test_getitem_single_string(self):
        frame = self.sample_spectra_frame()
        assert_series_equal(frame["A"], frame.data["A"])

    def test_setitem_single_string(self):
        frame = self.sample_spectra_frame()

        # Test existing data column
        frame["A"] = 5
        assert_array_equal(frame.data["A"].values, [5, 5, 5])

        # Test new data column
        frame["A_new"] = 1
        assert_array_equal(frame.data["A_new"].values, [1, 1, 1])

    # # Test cases for __setitem__
    def test_spectraframe_setitem(self):
        frame = self.sample_spectra_frame()

        # Set specific rows, columns, and wavelengths
        with pytest.raises(ValueError):
            frame[5, "A", 500] = 99.0
            frame[5, :, :] = 99.0

        # Set specific rows and wavelengths, all columns
        frame2 = frame.copy()
        frame2[6:8, :, 500] = 77.0
        assert np.array_equal(frame2.spc[1:3, 1], np.array([77.0, 77.0]))
        assert_frame_equal(frame2.data, frame.data)

        # Set specific rows, all columns, and a wavelength range
        frame2 = frame.copy()
        frame2[6:8, :, 400:600] = 88.0
        assert np.array_equal(frame2.spc[1:3, :], 88.0 * np.ones((2, 3)))
        assert_frame_equal(frame2.data, frame.data)

        # Set specific rows, specific columns, and all wavelengths
        frame2 = frame.copy()
        frame2[6:8, "A":"B", :] = 0
        assert_array_equal(frame2.spc, frame.spc)
        assert_array_equal(frame2.data.loc[6:8, "A":"B"].values, np.zeros((2, 2)))


class TestSpectraFrameAttrs:
    def sample_spectra_frame(self) -> SpectraFrame:
        spc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        wl = np.array([400, 500, 600])
        data = pd.DataFrame(
            {"A": [10, 11, 12], "B": [13, 14, 15], "C": [16, 17, 18]},
            index=[5, 6, 7],
        )
        return SpectraFrame(spc, wl, data)

    def test_getattr(self):
        frame = self.sample_spectra_frame()

        assert_series_equal(frame.A, frame.data.A)
        assert_index_equal(frame.index, frame.data.index)
        assert_index_equal(frame.columns, frame.data.columns)
        with pytest.raises(AttributeError):
            _ = frame.non_existent_attr


class TestSpectraFrameAssign:
    def sample_spectra_frame(self) -> SpectraFrame:
        spc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        wl = np.array([400, 500, 600])
        data = pd.DataFrame(
            {"A": [10, 11, 12], "B": [13, 14, 15]},
            index=[5, 6, 7],
        )
        return SpectraFrame(spc, wl, data)

    def test_assign_new_column(self):
        """Test assigning a new column to SpectraFrame"""
        frame = self.sample_spectra_frame()
        original_shape = frame.data.shape

        # Assign a new column
        result = frame.assign(C=[100, 200, 300])

        # Check that a new frame is returned (assign no longer modifies original)
        assert result is not frame
        assert frame.data.shape == original_shape
        assert "C" not in frame.data.columns

        # Check that the result has the new column
        assert result.data.shape == (original_shape[0], original_shape[1] + 1)
        assert "C" in result.data.columns
        assert_array_equal(result.data["C"].values, [100, 200, 300])

    def test_reassign_existing_column(self):
        """Test reassigning an existing column in SpectraFrame"""
        frame = self.sample_spectra_frame()
        original_shape = frame.data.shape
        original_a_values = frame.A.values.copy()

        # Assign to an existing column
        result = frame.assign(A=[100, 200, 300])

        # Check that a new frame is returned (assign no longer modifies original)
        assert result is not frame
        assert frame.data.shape == original_shape
        assert_array_equal(frame.A.values, original_a_values)

        # Check that the result has the modified column
        assert result.data.shape == original_shape
        assert_index_equal(result.data.columns, pd.Index(["A", "B"]))
        assert_array_equal(result.A.values, [100, 200, 300])

    def test_assign_multiple_columns(self):
        """Test assigning multiple new columns to SpectraFrame"""
        frame = self.sample_spectra_frame()

        # Assign multiple columns
        result = frame.assign(C=[100, 200, 300], D=["X", "Y", "Z"])

        # Check that original frame is unchanged
        assert "C" not in frame.data.columns
        assert "D" not in frame.data.columns

        # Check that result has the new columns
        assert "C" in result.data.columns
        assert "D" in result.data.columns
        assert_array_equal(result.data["C"].values, [100, 200, 300])
        assert_array_equal(result.data["D"].values, ["X", "Y", "Z"])

    def test_assign_spectral_data_unchanged(self):
        """Test that assign doesn't modify spectral data or wavelengths"""
        frame = self.sample_spectra_frame()
        original_spc = frame.spc.copy()
        original_wl = frame.wl.copy()

        frame.assign(new_col=[1, 2, 3])

        # Spectral data and wavelengths should remain unchanged
        assert_array_equal(frame.spc, original_spc)
        assert_array_equal(frame.wl, original_wl)


class TestSpectraFrameDrop:
    def sample_spectra_frame(self) -> SpectraFrame:
        spc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        wl = np.array([400, 500, 600])
        data = pd.DataFrame(
            {"A": [10, 11, 12], "B": [13, 14, 15], "C": [16, 17, 18]},
            index=[5, 6, 7],
        )
        return SpectraFrame(spc, wl, data)

    def test_drop_single_column(self):
        """Test dropping a single column from SpectraFrame"""
        frame = self.sample_spectra_frame()
        original_shape = frame.data.shape

        # Drop a single column
        result = frame.drop("C")

        # Check that the original frame is unchanged
        assert result is not frame
        assert frame.data.shape == original_shape
        assert "C" in frame.data.columns

        # Check that the result has the column dropped
        assert result.data.shape == (original_shape[0], original_shape[1] - 1)
        assert "C" not in result.data.columns
        assert "A" in result.data.columns
        assert "B" in result.data.columns

    def test_drop_multiple_columns(self):
        """Test dropping multiple columns from SpectraFrame"""
        frame = self.sample_spectra_frame()

        # Drop multiple columns
        result = frame.drop(["A", "C"])

        # Check original frame is unchanged
        assert "A" in frame.data.columns
        assert "C" in frame.data.columns

        # Check result has columns dropped
        assert "A" not in result.data.columns
        assert "C" not in result.data.columns
        assert "B" in result.data.columns
        assert result.data.shape[1] == 1

    def test_drop_all_columns(self):
        """Test dropping all columns from SpectraFrame"""
        frame = self.sample_spectra_frame()
        original_shape = frame.data.shape

        # Drop all columns
        result = frame.drop(["A", "B", "C"])

        # Check that the original frame is unchanged
        assert frame.data.shape == original_shape
        assert len(frame.data.columns) == 3

        # Check that the result has no columns
        assert result.data.shape[1] == 0
        assert len(result.data.columns) == 0

    def test_drop_spectral_data_unchanged(self):
        """Test that drop doesn't modify spectral data or wavelengths"""
        frame = self.sample_spectra_frame()
        original_spc = frame.spc.copy()
        original_wl = frame.wl.copy()

        result = frame.drop("A")

        # Original frame's spectral data and wavelengths should remain unchanged
        assert_array_equal(frame.spc, original_spc)
        assert_array_equal(frame.wl, original_wl)

        # Result's spectral data and wavelengths should be the same as original
        assert_array_equal(result.spc, original_spc)
        assert_array_equal(result.wl, original_wl)


class TestSpectraFrameApply:
    def sample_spectra_frame(self) -> SpectraFrame:
        spc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        wl = np.array([400, 500, 600])
        data = pd.DataFrame(
            {"A": [10, 11, 12], "B": [13, 14, 15], "C": [16, 17, 18]},
            index=[5, 6, 7],
        )
        return SpectraFrame(spc, wl, data)

    def test_string_function_scalar(self):
        frame = self.sample_spectra_frame()

        # Apply a NumPy function using a string
        result = frame.apply("sum", axis=0)
        assert np.array_equal(result.spc, np.sum(frame.spc, axis=0).reshape((1, -1)))
        assert np.array_equal(result.wl, frame.wl)
        assert result.data.shape == (1, 0)

        # Apply a NumPy function using a string with axis=1
        result = frame.apply("mean", axis=1)
        assert np.array_equal(result.spc, np.mean(frame.spc, axis=1).reshape((-1, 1)))
        assert np.array_equal(result.wl, [0])
        assert_frame_equal(result.data, frame.data)

    def test_string_function_vector(self):
        frame = self.sample_spectra_frame()
        q = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Apply a NumPy function using a string
        result = frame.apply("quantile", q, axis=0)
        assert np.array_equal(result.spc, np.quantile(frame.spc, q, axis=0))
        assert np.array_equal(result.wl, frame.wl)
        assert result.data.shape == (len(q), 0)

        # Apply a NumPy function using a string with axis=1
        result = frame.apply("quantile", q, axis=1)
        assert np.array_equal(result.spc, np.quantile(frame.spc, q, axis=1).T)
        assert np.array_equal(result.wl, np.arange(len(q)))
        assert_frame_equal(result.data, frame.data)

    def test_custom_function_scalar(self):
        frame = self.sample_spectra_frame()
        custom_function = lambda x: np.sum(x) * 2

        # Apply a custom callable function
        result = frame.apply(custom_function, axis=0)
        assert np.array_equal(result.spc, np.array([[24.0, 30.0, 36.0]]))
        assert np.array_equal(result.wl, frame.wl)
        assert result.data.shape == (1, 0)

        # Apply a custom callable function with axis=1
        result = frame.apply(custom_function, axis=1)
        assert np.array_equal(result.spc, np.array([[12.0], [30.0], [48.0]]))
        assert np.array_equal(result.wl, [0])
        assert_frame_equal(result.data, frame.data)

    def test_custom_function_vector(self):
        frame = self.sample_spectra_frame()
        custom_function = lambda x: [np.min(x), np.max(x)]

        # Apply a custom callable function
        result = frame.apply(custom_function, axis=0)
        assert np.array_equal(result.spc, np.array([[1, 2, 3], [7, 8, 9]]))
        assert np.array_equal(result.wl, frame.wl)
        assert result.data.shape == (2, 0)

        # Apply a custom callable function with axis=1
        result = frame.apply(custom_function, axis=1)
        assert np.array_equal(result.spc, np.array([[1, 3], [4, 6], [7, 9]]))
        assert np.array_equal(result.wl, [0, 1])
        assert_frame_equal(result.data, frame.data)

    def test_infer_wl(self):
        frame = self.sample_spectra_frame()
        custom_function = lambda x: x - np.min(x)

        # Apply a custom callable function with axis=1
        result = frame.apply(custom_function, axis=1)
        assert np.array_equal(result.wl, frame.wl)


class TestSpectraFrameApplyGroupBy:
    def sample_spectra_frame(self) -> SpectraFrame:
        return SpectraFrame(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
            ],
            wl=[400, 600, 800, 1000],
            data=pd.DataFrame(
                {"A": np.repeat([10, 11, 12], 2), "B": np.repeat(["B1", "B2"], 3)},
                index=list("abcdef"),
            ),
        )

    def test_scalar_by_single_column(self):
        frame = self.sample_spectra_frame()
        expected = SpectraFrame(
            frame.spc[[2, 5], :],
            wl=frame.wl,
            data=pd.DataFrame({"B": ["B1", "B2"], "group_index": [0, 0]}),
        )

        # Test string function
        result = frame.apply("max", groupby="B")
        assert_spectraframe_equal(result, expected)

        # Test custom function
        result = frame.apply(np.max, groupby="B")
        assert_spectraframe_equal(result, expected)

        # Test single column as a list
        result = frame.apply(np.max, groupby=["B"])
        assert_spectraframe_equal(result, expected)

    def test_scalar_by_multi_column(self):
        frame = self.sample_spectra_frame()
        expected = SpectraFrame(
            frame.spc[[1, 2, 3, 5], :],
            wl=frame.wl,
            data=pd.DataFrame(
                {
                    "B": ["B1", "B1", "B2", "B2"],
                    "A": np.array([10, 11, 11, 12], dtype=frame.A.dtype),
                    "group_index": [0, 0, 0, 0],
                }
            ),
        )

        # Test string function
        result = frame.apply("max", groupby=["B", "A"])
        assert_spectraframe_equal(result, expected)

        # Test custom function
        result = frame.apply(np.max, groupby=["B", "A"])
        assert_spectraframe_equal(result, expected)

    def test_vector_by_single_column(self):
        frame = self.sample_spectra_frame()
        expected = SpectraFrame(
            [
                0.5 * (frame.spc[0, :] + frame.spc[1, :]),
                0.5 * (frame.spc[1, :] + frame.spc[2, :]),
                0.5 * (frame.spc[3, :] + frame.spc[4, :]),
                0.5 * (frame.spc[4, :] + frame.spc[5, :]),
            ],
            wl=frame.wl,
            data=pd.DataFrame(
                {"B": np.repeat(["B1", "B2"], 2), "group_index": [0, 1, 0, 1]}
            ),
        )

        # Test string function
        result = frame.apply("quantile", [0.33, 0.67], groupby="B", method="midpoint")
        assert_spectraframe_equal(result, expected)

        # Test custom function
        result = frame.apply(np.quantile, [0.33, 0.67], groupby="B", method="midpoint")
        assert_spectraframe_equal(result, expected)

    def test_vector_by_multi_column(self):
        frame = self.sample_spectra_frame()
        expected = SpectraFrame(
            [
                0.5 * (frame.spc[0, :] + frame.spc[1, :]),
                0.5 * (frame.spc[0, :] + frame.spc[1, :]),
                frame.spc[2, :],
                frame.spc[2, :],
                frame.spc[3, :],
                frame.spc[3, :],
                0.5 * (frame.spc[4, :] + frame.spc[5, :]),
                0.5 * (frame.spc[4, :] + frame.spc[5, :]),
            ],
            wl=frame.wl,
            data=pd.DataFrame(
                {
                    "B": np.repeat(["B1", "B2"], 4),
                    "A": np.repeat([10, 11, 11, 12], 2).astype(frame.A.dtype),
                    "group_index": [0, 1] * 4,
                }
            ),
        )

        # Test string function
        result = frame.apply(
            "quantile", [0.33, 0.67], groupby=["B", "A"], method="midpoint"
        )
        assert_spectraframe_equal(result, expected)

        # Test custom function
        result = frame.apply(
            np.quantile, [0.33, 0.67], groupby=["B", "A"], method="midpoint"
        )
        assert_spectraframe_equal(result, expected)

    def test_infer_wl(self):
        frame = self.sample_spectra_frame()
        custom_function = lambda x: x - np.min(x)

        # Apply a custom callable function with axis=1
        result = frame.apply(custom_function, groupby=["B"], axis=1)
        assert np.array_equal(result.wl, frame.wl)


class TestSpectraFramePlot:
    def sample_spectra_frame(self) -> SpectraFrame:
        return SpectraFrame(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
            ],
            wl=[400, 600, 800, 1000],
            data=pd.DataFrame(
                {"A": np.repeat([10, 11, 12], 2), "B": np.repeat(["B1", "B2"], 3)},
                index=list("abcdef"),
            ),
        )

    def test_plots(self):
        frame = self.sample_spectra_frame()

        # Plot all
        frame.plot()

        # Plot in one row
        frame.plot(rows="B")
        frame.plot(rows=[1, 2, 3, 4, 5, 6])

        # Plot in one column
        frame.plot(columns="B")
        frame.plot(columns=[1, 2, 3, 4, 5, 6])

        # Plot 2d grid
        frame.plot(rows="B", columns="A")
        frame.plot(rows="B", columns=[1, 2, 3, 4, 5, 6])
        frame.plot(columns="B", rows=[1, 2, 3, 4, 5, 6])


class TestSpectraFrameMisc:
    def test_sample(self):
        sf = SpectraFrame(np.arange(5 * 3).reshape((5, 3)), data={"A": list("abcde")})
        np.random.seed(1)
        assert_spectraframe_equal(sf.sample(2), sf[[1, 2], :, :, True])


class TestSpectraFrameBaseline:
    def test_baseline(self):
        wl = 400 + np.arange(0, 10, 2)
        signal = np.array([0, 5, 10, 5, 0])
        bl = 10 + 5 * (wl - 400)
        sf = SpectraFrame(signal + bl, wl=wl)

        result = sf.baseline("rubberband")
        assert np.array_equal(result.spc[0, :], bl)

        result = sf.sbaseline("rubberband")
        assert np.array_equal(result.spc[0, :], signal)

    def test_baseline_threading(self):
        """Test that baseline correction with threading produces identical results."""
        wl = 400 + np.arange(0, 100, 2)
        # Create test spectra with baseline
        np.random.seed(42)
        n_spectra = 10
        signal = np.random.randn(n_spectra, len(wl))
        bl = 10 + 5 * (wl - 400)
        spectra_with_baseline = signal + bl
        
        sf = SpectraFrame(spectra_with_baseline, wl=wl)
        
        # Test that sequential and threaded processing give identical results
        baseline_sequential = sf.baseline("rubberband", n_jobs=1)
        baseline_threaded = sf.baseline("rubberband", n_jobs=4)
        
        # Results should be identical
        np.testing.assert_array_equal(baseline_sequential.spc, baseline_threaded.spc)
        np.testing.assert_array_equal(baseline_sequential.wl, baseline_threaded.wl)
        pd.testing.assert_frame_equal(baseline_sequential.data, baseline_threaded.data)
        
        # Test sbaseline as well
        sbaseline_sequential = sf.sbaseline("rubberband", n_jobs=1)
        sbaseline_threaded = sf.sbaseline("rubberband", n_jobs=4)
        
        np.testing.assert_array_equal(sbaseline_sequential.spc, sbaseline_threaded.spc)
        
    def test_baseline_threading_single_spectrum(self):
        """Test threading with single spectrum (edge case)."""
        wl = 400 + np.arange(0, 10, 2)
        signal = np.array([0, 5, 10, 5, 0])
        bl = 10 + 5 * (wl - 400)
        sf = SpectraFrame((signal + bl).reshape(1, -1), wl=wl)
        
        baseline_sequential = sf.baseline("rubberband", n_jobs=1)
        baseline_threaded = sf.baseline("rubberband", n_jobs=4)
        
        # Results should be identical even with single spectrum
        np.testing.assert_array_equal(baseline_sequential.spc, baseline_threaded.spc)


class TestSpectaFrameNormalize:
    def test_normalize(self):
        np.random.seed(321)
        sf = SpectraFrame(
            np.random.randn(5 * 10).reshape((5, -1)),
            wl=np.arange(600, 1600, 100),
            data={"A": list("abcde")},
        )

        result = sf.normalize("01")
        np.testing.assert_almost_equal(np.min(result.spc, axis=1), 0)
        np.testing.assert_almost_equal(np.max(result.spc, axis=1), 1)

        result = sf.normalize("mean")
        np.testing.assert_almost_equal(np.mean(result.spc, axis=1), 1)

        result = sf.normalize("vector")
        np.testing.assert_almost_equal(np.sum(np.power(result.spc, 2), axis=1), 1)

        result = sf.normalize("peak")
        np.testing.assert_almost_equal(np.max(result.spc, axis=1), 1)

        result = sf.normalize("peak", peak_range=(800, 1200))
        np.testing.assert_almost_equal(np.max(result[:, :, 800:1200].spc, axis=1), 1)


class TestSpectraFrameFromFile:
    def sample_spectra_frame(self) -> SpectraFrame:
        # Create a dummy SpectraFrame
        sf = SpectraFrame(
            spc=np.array([[1, 2, 3], [4, 5, 6]]),
            wl=np.array([400, 500, 600]),
            data=pd.DataFrame(
                {
                    "sample": ["A", "B"],
                    "type": ["X", "Y"],
                }
            ),
        )
        return sf

    def test_csv(self, tmp_path):
        sf = self.sample_spectra_frame()
        out_path: Path = tmp_path / "test.csv"

        sf.to_pandas(multiindex=False).to_csv(out_path, index=False)
        assert out_path.exists()

        sf_imported = SpectraFrame.fromfile(out_path)

        assert_array_equal(sf_imported.wl, sf.wl)
        assert_array_equal(sf_imported.spc, sf.spc)
        assert_frame_equal(sf_imported.data, sf.data)

    def test_pickle(self, tmp_path):
        sf = self.sample_spectra_frame()
        multi_index = [False, True]
        string_names = [False, True]

        for mi in multi_index:
            for sn in string_names:
                out_path: Path = tmp_path / f"test_{sn}_{mi}.pkl"

                # Export to a temporary file
                sf.to_pandas(string_names=sn, multiindex=mi).to_pickle(out_path)
                assert out_path.exists()

                # Read with fromfile
                sf_imported = SpectraFrame.fromfile(out_path)

                # Verify correct loading
                assert_array_equal(sf_imported.wl, sf.wl)
                assert_array_equal(sf_imported.spc, sf.spc)
                assert_frame_equal(sf_imported.data, sf.data)
