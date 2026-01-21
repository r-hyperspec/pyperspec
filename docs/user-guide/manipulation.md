# Data Manipulation

This guide covers advanced techniques for manipulating spectroscopic data.

## Filtering and Selection

### Query-based Filtering

```python
# Simple conditions
control_samples = sf.query("group == 'Control'")
high_conc = sf.query("concentration > 2.0")

# Complex conditions with multiple criteria
filtered = sf.query("group == 'Treatment' and concentration > 1.5 and pH < 7.0")

# Using isin for multiple values
selected_samples = sf.query("sample_id.isin(['S1', 'S3', 'S5'])")
```

### Wavelength Range Selection
```python
# Select fingerprint region (Raman)
fingerprint = sf[:, :, 600:1800]

# Select specific wavelengths
peaks = sf[:, :, [1003, 1458, 1618]]
```

### Random Sampling
```python
# Sample random spectra
sample_subset = sf.sample(n=10)

# Sample with replacement
bootstrap_sample = sf.sample(n=100, replace=True)
```

## Data Aggregation

### Groupwise Operations

```python
# Mean spectra by group
group_means = sf.mean(groupby='group')

# Multiple grouping variables
condition_means = sf.mean(groupby=['group', 'time_point'])

# Other statistics
group_std = sf.std(groupby='group')
group_median = sf.median(groupby='group')

# Multi-output aggregation
group_quantiles = sf.quantile(groupby='group', q=[0.25, 0.5, 0.75])
```

### Custom Aggregation Functions

```python
my_custom_function = lambda x: np.mean(x) / np.std(x)
sf_custom = sf.apply(my_custom_function, groupby='group', axis=1)

# Apply numpy functions
sf_trapz = sf[:, :, 1000:1100].apply(np.trapz, axis=1)

# Maximum intensity
max_intensity = sf.max(axis=1)
```

## Data Transformation

### Mathematical Operations
```python
# Arithmetic operations
sf_log = sf.apply(np.log1p, axis=1)  # Log transform
sf_sqrt = sf.apply(np.sqrt, axis=1)  # Square root

# Element-wise operations between SpectraFrames
sf_ratio = sf1 / sf2
sf_difference = sf1 - sf2

# Operations with scalars
sf_scaled = sf * 1000
sf_offset = sf + 0.1
```

### Spectral Derivatives

```python
# First derivative
first_deriv = sf.apply(np.diff, axis=1)

# Second derivative using scipy
from scipy import ndimage
second_deriv = sf.apply(lambda x: ndimage.gaussian_filter1d(x, sigma=1, order=2), axis=1)
```

## Metadata Manipulation

### Adding and Removing Columns
```python
# Simple assignment
sf['new_column'] = ['A', 'B', 'C', 'A', 'B']
# Same but using assign. This is useful for chaining operations.
sf = sf.assign(new_column=lambda x: x['group'] + '_new')
# Similarly, we can remove columns
sf = sf.drop(columns=['new_column'])

# Calculated columns
sf['total_intensity'] = sf.spc.sum(axis=1)
sf['peak_ratio'] = sf[:, :, 1450].spc.flatten() / sf[:, :, 1000].spc.flatten()

# Conditional columns
sf['high_intensity'] = sf['total_intensity'] > sf['total_intensity'].median()
```

### Metadata from External Sources
```python
# Merge with external data
import pandas as pd
external_data = pd.read_csv('sample_info.csv')
sf.data = sf.data.merge(external_data, on='sample_id', how='left')

# Update existing columns
sf.data.loc[sf.data['group'] == 'Control', 'treatment'] = 'None'
```

## Sorting

You can sort spectra by row index or by metadata values, similar to pandas.
Wavelengths can be sorted independently with `wl_sort`, which also reorders
the spectral columns to match the new wavelength order.

```python
sf_sorted = sf.sort_index()
sf_sorted = sf.sort_values("group")
sf_sorted = sf.wl_sort()
```

## Concatenation
```python
# Combine multiple SpectraFrames
combined = pyspc.concat([sf1, sf2, sf3])

# Ensure consistent wavelength grids before concatenation
sf2_aligned = sf2.wl_resample(sf1.wl)
combined = pyspc.concat([sf1, sf2_aligned])
```

Concat along the wavelength axis can be used to cut out a specific region

```python
wl = np.linspace(400, 800, 100)
sf = pyspc.SpectraFrame(np.random.rand(10, 100), wl=wl)

sf_cutted = pyspc.concat([
    sf[:,:, :500],  # Select wavelengths 400-500 1/cm
    sf[:,[], 700:],  # Select wavelengths 700-800 1/cm
], axis=1)
```


## Missing Data Handling

### Identifying Missing Data
```python
# Check for NaN values
has_nan = np.isnan(sf.spc).any(axis=1)
nan_spectra = sf[has_nan, :, :]

# Missing metadata
missing_metadata = sf.data.isnull().sum()
```

### Handling Missing Values

```python
# Fill NaN values using linear interpolation
spc = spc.apply(pyspc.utils.fillna, axis=1)
```

## Converting to Other Formats

```python
# Convert to pandas DataFrame
df = sf.to_pandas()
# Export to CSV
df.to_csv('spectra_data.csv', index=False)

# Convert to numpy array
spectra_array = sf.spc
# Or alternatively
spectra_array = np.array(sf)
```

## See Also

- [Preprocessing Methods](preprocessing.md) for spectral preprocessing
- [Visualization](visualization.md) for plotting techniques
