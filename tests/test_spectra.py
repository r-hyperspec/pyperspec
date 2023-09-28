import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
import pytest

from pyspc import SpectraFrame


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
        elif side == "rigth":
            sf_res = op(10, sf_res)
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


class TestSpectraFrameSlicers:
    def sample_spectra_frame(self) -> SpectraFrame:
        spc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        wl = np.array([400, 500, 600])
        data = pd.DataFrame(
            {"A": [10, 11, 12], "B": [13, 14, 15], "C": [16, 17, 18]},
            index=[5, 6, 7],
        )
        return SpectraFrame(spc, wl, data)

    # Test cases for __getitem__
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

    # Test cases for __getitem__
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

    # # Test cases for __setitem__
    # def test_spectraframe_setitem(self):
    #     frame = self.sample_spectra_frame()

    #     # Set specific rows, columns, and wavelengths
    #     frame[0, "A", 500] = 99.0
    #     assert frame.spc[0, 1] == 99.0
    #     assert frame.data.loc[0, "A"] == 99.0

    #     # Set specific rows and wavelengths, all columns
    #     frame[1:3, :, 500] = 77.0
    #     assert np.array_equal(frame.spc[1:3, 1], np.array([77.0, 77.0]))
    #     assert np.array_equal(frame.data.loc[1:2, "A"], np.array([77.0, 77.0]))

    #     # Set specific rows, all columns, and a wavelength range
    #     frame[1:3, :, 400:600] = 88.0
    #     assert np.array_equal(
    #         frame.spc[1:3, :], np.array([[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    #     )
    #     assert np.array_equal(
    #         frame.data.loc[1:2, :], np.array([[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]])
    #     )


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
