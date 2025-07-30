# Quick Start Guide

This guide will get you up and running with PyPerSpec in minutes.

## Your First SpectraFrame

```python
import pyspc
import numpy as np
import pandas as pd

# Create some sample spectral data
np.random.seed(42)
spectra = np.random.rand(5, 100)  # 5 spectra, 100 wavelength points
wavelengths = np.linspace(400, 4000, 100)  # Wavenumbers from 400 to 4000 cm⁻¹
metadata = pd.DataFrame({
    'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
    'group': ['Control', 'Control', 'Treatment', 'Treatment', 'Control'],
    'concentration': [1.0, 1.5, 2.0, 2.5, 1.2]
})

# Create a SpectraFrame object
sf = pyspc.SpectraFrame(spectra, wl=wavelengths, data=metadata)
print(sf)
```

## Basic Operations

### Accessing Data
```python
# Access metadata columns like a DataFrame
print(sf.concentration)
print(sf['sample_id'])

# Access spectral data for specific wavelengths
print(sf[:, :, 1000:2000])  # Wavelength range 1000-2000

# Filter by metadata
control_spectra = sf.query("group == 'Control'")
print(f"Control samples: {control_spectra.nspc}")
```

### Data Processing

```python
# Normalize spectra
sf_normalized = sf.normalize('area')

# Calculate baseline
bl = sf.baseline('rubberband')
# Apply baseline correction
sf_baseline = sf.sbaseline('rubberband')

# Smooth spectra
sf_smooth = sf.smooth('savgol', window_length=5, polyorder=2)

# Chain operations
sf_processed = (sf
    .baseline('rubberband')
    .smooth('savgol', window_length=5, polyorder=2)
    .normalize('area')
)
```

### Statistical Operations

```python
# Calculate mean spectra by group
mean_by_group = sf.mean(groupby='group')

# Calculate statistics
sf_stats = sf.std(groupby='group')

# Apply custom functions
sf_custom = sf.apply(lambda x: np.log(x + 1), axis=1)
```

### Visualization

```python
# Plot all spectra
sf.plot()

# Plot by groups
sf.plot(colors='group')

# Plot with custom layout
sf.plot(rows='group', colors='sample_id')
```

## Loading Real Data

```python
# Convert to pandas DataFrame for analysis
df = sf.to_pandas()
df.to_csv('spectra_data.csv', index=False)

# Load from a file (example format)
sf_from_file = pyspc.SpectraFrame.fromfile('spectra_data.csv')
```

## Next Steps

- Read the [Basic Concepts](concepts.md) to understand the data structure
