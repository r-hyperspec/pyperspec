# Preprocessing

This guide covers the various preprocessing operations available in `pyperspec` for spectral data preprocessing.
Preprocessing methods are applied to `SpectraFrame` objects, which contain spectral data and metadata.
These methods can be chained together to create complex preprocessing pipelines.

???+ info "Basic Principle"
    Methods of `SpectraFrame` are designed as a bridge to the rich ecosystem of spectral processing libraries, such as `pybaselines`, `scipy`, and `numpy`. For our custom implementations, we split the functionality into separate modules (e.g., `spikes`, `peaks`) that are independent from `SpectraFrame`.

## Baseline Correction

Baseline correction removes systematic drift and background signals from spectra. 
[pybaselines](https://pybaselines.readthedocs.io/en/latest/) is a rich package of baseline correction methods. Basically, `SpectraFrame.baseline()` method provides a bridge to the `pybaselines` methods.
General interface is `SpectraFrame.baseline(method, **kwargs)`, where `method` is the baseline correction algorithm and `**kwargs` are method-specific parameters.
Full list of available methods and parameters can be found in the [pybaselines docs](https://pybaselines.readthedocs.io/en/latest/).
Here are some examples:

```python
import pyspc

# Load your spectral data
sf = pyspc.SpectraFrame(spectra_data, wl=wavelengths)

# Rubberband baseline correction
baseline_rb = sf.baseline("rubberband")

# AIRPLS (Adaptive Iteratively Reweighted Penalized Least Squares)
baseline_airpls = sf.baseline("airpls", lam=1e5)

# SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
baseline_snip = sf.baseline("snip", max_half_window=40)
```

To subtract the baseline from your spectra, you can use the `-` operator:

```python
# Subtract baseline using - operator
corrected_spectra = sf - baseline_snip
```

Or, for convenience, use `sbaseline()` (short for `subtract_baseline()`) to directly subtract the baseline from your spectra:

```python
# Equivalent to the above subtraction
corrected_spectra = sf.sbaseline("airpls", lam=1e5)
```

It is recommended to test baseline correction methods on a subset of your data before applying them to the entire dataset.

```python
# Test baseline removal
spc_ = spc.sample(10)
# Smoothing spectra before baseline correction can help to improve the results.
spc_sm_ = spc_.smooth("savgol", window_length=11, polyorder=2)
# Apply baseline correction
# We can compare different methods on the same spectra
bl_airpls = spc_sm_.baseline(method="airpls", lam=1e5)
bl_rubberband = spc_sm_.baseline(method="rubberband")
bl_snip = spc_sm_.baseline(method="snip", max_half_window=40)

# Plot for random subsets
fig, ax = plt.subplots()
ax.set_title("Baseline Correction Comparison")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.plot(spc_.wl, spc_.spc.T, color="black", alpha=0.5, label="Original")
ax.plot(spc_.wl, spc_sm_.spc.T, color="gray", alpha=0.5, label="Smoothed")
ax.plot(spc_.wl, bl_airpls.spc.T, label="AIRPLS", color="blue")
ax.plot(spc_.wl, bl_rubberband.spc.T, label="Rubberband", color="orange")
ax.plot(spc_.wl, bl_snip.spc.T, label="SNIP", color="green")
ax.legend()
plt.show()

# Finally use the best method on the entire dataset
# For example, let's use rubberband method
spc_nobl = spc.smooth("savgol", window_length=11, polyorder=2).sbaseline(method="rubberband")

# Plot after baseline removal
spc_nobl.sample(10).plot(colors="index")
```

## Normalization

Normalization standardizes spectral intensities to enable meaningful comparisons between spectra. Several normalization methods are available:

```python
# Min-Max normalization (0-1 scaling)
sf_norm = sf.normalize("01")

# Area normalization
sf_norm = sf.normalize("area")

# Vector normalization (L2 norm)
sf_norm = sf.normalize("vector")

# Mean normalization
sf_norm = sf.normalize("mean")

# Peak normalization (normalize by maximum peak in range)
sf_norm = sf.normalize("peak", peak_range=(1000, 1200))
# By default, peak value is approximated by the maximum value in the given range.
# To use a different method, use the `**kwargs` to pass to `around_max_peak_fit` function.
sf_norm = sf.normalize("peak", peak_range=(1000, 1200), method="gaussian", window=5)

# For custom normalization you can use `/` directly:
sf_norm = sf / sf.median(axis=1) 
```

## Smoothing

Smoothing reduces noise in spectral data while preserving important spectral features.
Currently, pyperspec supports Savitzky-Golay filtering.
The parameters are passed directly to `scipy.signal.savgol_filter()`.

```python
# Basic Savitzky-Golay smoothing
sf_smooth = sf.smooth("savgol", window_length=5, polyorder=3)

# With custom parameters
sf_smooth = sf.smooth("savgol", window_length=11, polyorder=2, mode='nearest')
```

Alternatively, `.baseline()` can be used as some methods from `pybaseline` apply smoothing.
Another option is to use `sf.apply(...)` to apply any custom smoothing function.

## Wavelengths/Wavenumbers Resampling

Resampling allows you to interpolate spectral data onto a new wavelength grid, useful for:
- Standardizing wavelength ranges across different instruments/measureements
- Increasing or decreasing spectral resolution
- Aligning spectra for analysis

The method passes data to [`scipy.interpolate.interp1d`](https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.interpolate.interp1d.html) for interpolation.


```python
# Define new wavelength grid
new_wavelengths = np.linspace(400, 1000, 500)

# Resample spectra
sf_resampled = sf.wl_resample(new_wavelengths)

# With custom interpolation parameters
sf_resampled = sf.wl_resample(new_wavelengths, 
                             kind='cubic', 
                             bounds_error=False, 
                             fill_value='extrapolate')
```

## Spike Removal

???+ note "NOTE: not a method of `SpectraFrame`"
    `find_spikes` is not a method of `SpectraFrame`, but a standalone function in the `spikes` module that operates on numpy arrays. Use it by passing your spectral data array directly.

Cosmic ray spikes and other artifacts can be detected and removed from spectral data. pyperspec provides the `find_spikes()` function from the `spikes` module.

General idea of all spike detection methods is to identify outliers in the spectral data based on statistical properties. It is common to use derivatives. Find more about the parameters in the [spikes API](../api-spikes.md).

Here is an example of how to use `find_spikes()`:

```python
spikes = pyspc.spikes.find_spikes(spc.spc, ndiff=1, method="zscore", threshold=10)
is_spiky = np.any(spikes, axis=1)

# Plot the spikes
spc[is_spiky, :, :].sample(10).plot(colors="index")

# Fill the spikes with NaN values
spc.spc[spikes] = np.nan

# Fill NaN values using linear interpolation
spc = spc.apply(pyspc.utils.fillna, axis=1)

# See the results
spc[is_spiky, :, :].sample(10).plot(colors="index")
```

## Custom Preprocessing

You can also apply custom preprocessing functions to your spectra using the `apply()` method. This allows you to use any function that operates on numpy arrays:

```python
def custom_preprocessing(spectrum):
    # Example custom function that applies a simple transformation
    return spectrum * 2  # Just an example, replace with your logic

# Apply custom preprocessing
sf_custom = sf.apply(custom_preprocessing, axis=1)
```

## Cutting out Specific Regions
You can cut out specific regions of your spectra by concatenating slices along the wavelength axis. This is useful for focusing on specific spectral features or removing unwanted regions.

```python
# Cut out Raman silent region 1800-2700 cm⁻¹
import pyspc
sf_cutted = pyspc.concat([
    sf[:, :, :1800],  # Select wavenumbers till 1800 cm⁻¹
    sf[:, [], 2700:],  # Select wavenumbers from 2700 cm⁻¹
], axis=1)
```

## Correlation and Spectral Matching

```python
# Calculate correlation with reference spectrum
reference = sf.query("sample_id == 'REF001'").spc[0, :]

def correlation_with_ref(spectrum):
    return np.corrcoef(spectrum, reference)[0, 1]

sf['correlation'] = sf.apply(correlation_with_ref, axis=1).spc.flatten()
best_matches = sf.query("correlation > 0.95")
```

## Piping and Chaining

`SpectraFrame` supports method chaining, allowing you to combine multiple preprocessing steps elegantly (as you would do with pandas DataFrames):

```python
# More complex preprocessing pipeline
processed_sf = (sf
    .wl_resample(np.linspace(400, 4000, 1000))
    .smooth("savgol", window_length=7, polyorder=2)
    .sbaseline("snip", max_half_window=40)
    .normalize("peak", peak_range=(2800, 3000))
)
```

## Best Practices

**Order**: Generally follow this sequence:

   - Resampling (if needed)
   - Smoothing (to reduce noise before baseline correction)
   - Baseline correction
   - Normalization (for classification or comparison purposes)

**Parameter optimization**: Test different parameters on representative spectra before applying to entire datasets
