# Basic Concepts

Understanding the core concepts of `pyperspec` will help you work more effectively with spectroscopic data.



## The SpectraFrame Object

The `SpectraFrame` is the central data structure in `pyperspec`. It combines three essential components:

```
SpectraFrame Structure:
                                     sf.data
          ┌─────────────────────┐    ↓
 sf.wl →  │  w1  w2  w3 ...  wm │  ┌────────────────────┐
          └─────────────────────┘  | col1 col2 ... colm │
          ┌─────────────────────┐  ├────────────────────┤ 
sf.spc →  │ x11 x12 x13 ... x1m │  │ ...  ...  ...  ... │
          │ x21 x22 x23 ... x2m │  │ ...  ...  ...  ... │
          │ ...                 │  │ ...  ...  ...  ... │
          │ xn1 xn2 xn3 ... xnm │  │ ...  ...  ...  ... │
          └─────────────────────┘  └────────────────────┘
```

| Component         | Type                | Shape                        | Content                                         | Access                                  |
|-------------------|---------------------|------------------------------|-------------------------------------------------|-----------------------------------------|
| **Spectral Data (`spc`)**   | 2D numpy array      | (n_spectra, n_wavelengths)         | Intensity values for each spectrum at each wavelength | `sf.spc`                                |
| **Wavelength/Wavenumber Array (`wl`)** | 1D numpy array      | (n_wavelengths,)                  | Wavelength or wavenumber values                 | `sf.wl`                                 |
| **Metadata (`data`)**       | pandas DataFrame    | (n_spectra, n_metadata_columns)     | Sample information, experimental conditions, etc. | `sf.data`, `sf.column_name`, `sf['column_name']` |


This structure works for various types of spectroscopic data, for example:

- **Raman**: Wavenumber (cm⁻¹) vs Intensity
- **FTIR**: Wavenumber (cm⁻¹) vs Absorbance/Transmittance  
- **UV-Vis**: Wavelength (nm) vs Absorbance
- **Fluorescence**: Wavelength (nm) vs Fluorescence Intensity
- **NIR**: Wavelength (nm) vs Reflectance/Absorbance
- **XRF**: Energy (keV) vs Counts
- **Mass Spectrometry**: m/z vs Intensity


## Key Properties

```python
# Shape information
sf.shape           # (n_rows, n_cols, n_wavelengths)
sf.nspc            # Number of spectra
sf.nwl             # Number of wavelength points

# Data access
sf.index           # Row indices (same as sf.data.index)
sf.columns         # Metadata column names (same as sf.data.columns)
sf.is_equally_spaced  # Whether wavelengths are equally spaced
```

## Indexing and Slicing

`pyperspec` uses a three-dimensional indexing system: `[<index>, <data columns>, <wavelengths>]`
This simulates the behavior of `hyperSpec` and allows for flexible data manipulation.
By default, `.loc` style indexing is used, but you can also use `.iloc`-style indexing for more flexibility: `[<index>, <data columns>, <wavelengths>, True]`

Basically, `sf[a, b, c]` is equivalent to `SpectraFrame(sf.spc[a,c], sf.wl[c], sf.data.loc[a, b])`, and `sf[a,b,c,True]` is equivalent to `SpectraFrame(sf.spc[a,c], sf.wl[c], sf.data.iloc[a, b])` with some additional checks.


For example, 

```python
# Select wavelength ranges  
sf[:, :, 1000:2000]  # Wavelength range 1000-2000
sf[:, :, [1000, 1500, 2000]]  # Specific wavelengths

# Select metadata columns
sf[:, ['group', 'concentration'], :]

# Use the 4th parameter as True for iloc-style indexing
sf[:3, :2, :5, True]  # First 3 spectra, 2 metadata cols, 5 wavelengths
```

You can find more on this in [Data Manipulation](user-guide/manipulation.md) guide.

## Dispatching to components

By default, `SpectraFrame` tries to orchestrate between the spectral data, metadata, and wavelengths. I.e. the main role is provide a convinient proxy to the underlying components rather that implementing all the algorithms. In most cases, methods/operations are just dispatched and corresponding `SpectraFrame` is constructed.

Basic principles:

1. All arithmetic operations (e.g. `+`, `-`, `*`, `/`) are passed to the `spc` component, e.g., `sf + x` is equivalent to `SpectraFrame(sf.spc + x, sf.wl, sf.data)`.
2. Statistics/aggregation methods (e.g. `mean`, `std`, `sum`) are passed to the `spc` component, e.g., `sf.mean()` computes the mean spectrum, however the metadata can be used to group the results, e.g., `sf.mean(groupby='group')`.
3. Metadata operations (sorting, filtering, querying, adding/removing columns) (TBD: currently only `query` is available) are passed to the `data` component, e.g., `sf.query("group == 'Control'")` is equivalent to `SpectraFrame(sf.spc[sf['group']=='Control'], sf.wl, sf.data[sf['group']=='Control'])`.
4. Preprocessing methods (e.g. `normalize`, `baseline`, `smooth`) are applied to the `spc`, e.g., `sf.normalize('area')` applies area normalization to the spectral data and returns a new `SpectraFrame` with the same wavelengths and metadata. Methods like `smooth`, `baseline` are just passing data to the corresponding algorithms in `pybaselines` or `scipy.signal` and returning a new `SpectraFrame` with the processed data.


```python
# Get only spectra of 'Control' group
sf.query("group == 'Control'")

# Apply arithmetic operations
bl = sf.baseline('rubberband')
sf_nobaseline = sf - bl
```

More examples and details can be found in [Data Manipulation](user-guide/manipulation.md) and [Preprocessing](user-guide/preprocessing.md) guides.

## Next Steps

- Learn about [Data Manipulation](user-guide/manipulation.md) techniques
- Explore [Preprocessing](user-guide/preprocessing.md) for spectral preprocessing
