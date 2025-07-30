# Import/Export

Exporting `SpectraFrame` objects works via convering them to pandas DataFrames from where you can export to any format available in pandas, such as CSV, pickle, feather, parquet, etc.

When converting to pandas DataFrame, various options are available to help with following exports/imports:

- `multiindex`: If `True`, the DataFrame will have a MultiIndex with the metadata columns as levels. This is useful for quickly split spectra data and metadata.
- `string_names`: If `True`, the metadata column names will be converted to strings. This is useful for some export formats that do not support non-string column names.

```python
import pyspc

# Create sample data
wl = np.linspace(400, 800, 100)
spectra = np.random.randn(10, 100)
sf = pyspc.SpectraFrame(spectra, wl=wl)

# Some export examples
sf.to_pandas().to_csv('spectra.csv', index=False)  # Export to CSV
sf.to_pandas(multiindex=True).to_pickle('spectra.pkl')  # Export to pickle
sf.to_pandas(string_names=True).to_feather('spectra.feather')  # Export to Feather
sf.to_pandas(string_names=True).to_parquet('spectra.parquet')  # Export to Parquet
```

When exported directly from `SpectraFrame`, it is possible to import the data directly from that file:

```python
sf = pyspc.SpectraFrame.fromfile('spectra.csv')
sf = pyspc.SpectraFrame.fromfile('spectra.pkl')
sf = pyspc.SpectraFrame.fromfile('spectra.feather')
sf = pyspc.SpectraFrame.fromfile('spectra.parquet')
```

!!! note
    When taken from external sources, the data can be has to be converted to `SpectraFrame` explicitly.
