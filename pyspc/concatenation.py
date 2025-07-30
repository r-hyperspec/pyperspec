from typing import Iterable

import numpy as np
import pandas as pd

from .spectra import SpectraFrame


def concat(objs: Iterable[SpectraFrame], axis=None) -> SpectraFrame:
    """Concatenate multiple `SpectraFrame` objects along a specified axis.

    Parameters
    ----------
    objs : Iterable[SpectraFrame]
        An iterable of `SpectraFrame` objects to concatenate.
    axis : int, optional
        The axis along which to concatenate the objects. If `None`, it will be inferred.
        If all objects have the same `nspc` and different `wl`, it will default to `1`
        (concatenate by columns).  If all objects have the same wavelengths, it will
        default to `0` (concatenate by rows).

    Returns
    -------
    SpectraFrame
        A new `SpectraFrame` object containing the concatenated spectral data.

    Raises
    ------
    ValueError
        If the input `SpectraFrame` objects have incompatible shapes or wavelengths.

    Notes
    -----
    - indices of the `data` DataFrame are reset after concatenation.
    - No additional checks and sortings are performed. Pay attention to:
        order of `wl` and overlapping ranges of `wl`
    """
    # TODO: Allow concatenating np.ndarray and pd.DataFrame
    is_same_wls = len(set(tuple(sf.wl.astype(np.float64)) for sf in objs)) == 1
    is_same_nspc = len(set(sf.nspc for sf in objs)) == 1

    if axis is None:
        # If all objects have the same number of rows and different number of
        # wavelengths then assume that we want to stack by columns
        if (not is_same_wls) and is_same_nspc:
            axis = 1
        else:
            axis = 0

    if axis == 0:
        if is_same_wls:
            return SpectraFrame(
                np.concatenate([sf.spc for sf in objs], axis=0),
                wl=objs[0].wl,
                data=pd.concat([sf.data for sf in objs], axis=0, ignore_index=True),
            )
        else:
            raise ValueError("Spectral data has different wavelenghts")
    elif axis == 1:
        # Check overlapping data columns
        data_columns = pd.Series(
            [col for sf in objs for col in sf.data.columns]
        ).value_counts()
        if any(data_columns > 1):
            duplicates = data_columns[data_columns > 1].index.tolist()
            raise ValueError(f"Overlapping data columns found: {duplicates}")

        # TODO: Order of wl
        # TODO: Check that wl ranges do not overlap
        return SpectraFrame(
            np.concatenate([sf.spc for sf in objs], axis=1),
            wl=np.concatenate([sf.wl for sf in objs], axis=None),
            data=pd.concat(
                [sf.data.reset_index(drop=True) for sf in objs],
                axis=1,
                ignore_index=False,
            ),
        )
    else:
        raise ValueError(f"Unexpected `axis` {axis}.")
