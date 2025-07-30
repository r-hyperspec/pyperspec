[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable release yet.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


# Python Package pyperspec

This is a Python package designed to simplify the analysis and manipulation of hyperspectral datasets.
The package provides an object-oriented approach providing a user-friendly interface that feels familiar to Python users, i.e. close to libraries such as `numpy`, `pandas`, and `scikit-learn`.

This is heavily inspired by R package [hyperSpec](https://github.com/r-hyperspec/hyperSpec) and part of **`r-hyperspec`**.
The goal is to make the work with hyperspectral data sets, (i.e. spatially or time-resolved spectra, or spectra with any other kind of information associated with each of the spectra) more comfortable.
The spectra can be data obtained during 
[XRF](https://en.wikipedia.org/wiki/X-ray_fluorescence),
[UV/VIS](https://en.wikipedia.org/wiki/Ultraviolet%E2%80%93visible_spectroscopy), 
[Fluorescence](https://en.wikipedia.org/wiki/Fluorescence_spectroscopy),
[AES](https://en.wikipedia.org/wiki/Auger_electron_spectroscopy),
[NIR](https://en.wikipedia.org/wiki/Near-infrared_spectroscopy),
[IR](https://en.wikipedia.org/wiki/Infrared_spectroscopy), 
[Raman](https://en.wikipedia.org/wiki/Raman_spectroscopy), 
[NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy), 
[MS](https://en.wikipedia.org/wiki/Mass_spectrometry),
etc. spectroscopy measurements.

**NOTE:** The main focus is **not** on algroithms since there are already many other good packages implementing algorithms, e.g. 
[numpy](https://numpy.org/),
[scipy](https://scipy.org/),
[pybaselines](https://github.com/derb12/pybaselines),
and more can be found in [FOSS For Spectroscopy](https://bryanhanson.github.io/FOSS4Spectroscopy/) list.

Rather, it provides convinient interface for those algorithms and other routine tasks.

For detailed information and documentation, please visit [Documentation](https://r-hyperspec.github.io/pyperspec/).

## Documentation

Please, check [here](https://r-hyperspec.github.io/pyperspec/)

## Installation

Currently available only from GitHub:

```bash
pip install git+https://github.com/r-hyperspec/pyperspec.git
```

## Quick Demo

```python
import pyspc
import numpy as np
import pandas as pd

spc = np.random.rand(10, 20) # Here are your spectra in unfolded structure
wl = np.linspace(1000,2000,20) # Array of wavelength/wavenumbers
meta_data = pd.DataFrame({"group": ["A", "B"] * 5, "date": pd.date_range("2023-01-01", periods=10)}) # Additional meta-data

# Create the object
sf = pyspc.SpectraFrame(spc, wl=wl, data=meta_data)

# Easy meta-data manipulation
sf.A
sf["A"]
sf["group"] = ["Control", "Treatment"] * 5

# Easy data slicing/filtering, similar to hyperSpec
sf[:,:,500:1000] # Cut wavelength range to [500, 1000]
sf[:5,:,:5, True] # Use iloc style to get only first five spectra and first five wavenumbers
sf.query("group == 'Control'") # Get only 'Control' group

# Simple aggregation even with custom methods
sf[:,:,500:1000].mean(groupby=["group", "date"])
sf.query("group == 'Control'").apply(lambda x: np.sum(x**2), axis=0)

# Chaining methods
sf_processed = (
    sf.query("group == 'Control'")
    .mean(groupby="date")
    .smooth("savgol", window_length=7, polyorder=2)
    .sbaseline("rubberband")
    .normalize("area")
)

# Select 3 random spectra and plot them colored by "date"
sf.sample(3).plot(colors="date")

# Export to wide pandas DataFrame
sf.to_pandas()
```

## Acknowlegments

* This project was a continuation of [ibcp/pyspectra](https://github.com/ibcp/pyspectra). We acknowlege support and contribution of [Emanuel Institute of Biochemical Physics, RAS](https://biochemphysics.ru/)
* This project has received funding from the European Union’s [Horizon 2020](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en) research and innovation programme under the [Marie Sklodowska-Curie Actions](https://marie-sklodowska-curie-actions.ec.europa.eu/) *(Grant Agreement 861122)* as part of [IMAGE-IN](https://image-in-itn.eu/) project.
* The project was developed as part of secondment at [Chemometrix GmbH](https://chemometrix.gmbh/)
* Supervision was from [Chemometrix GmbH](https://chemometrix.gmbh/), [Leibniz-IPHT](https://www.leibniz-ipht.de/en/), and [BMD Software](https://www.bmd-software.com/)

<img src="https://biochemphysics.ru/static/img/logo_en.png" alt="Emanuel Institute of Biochemical Physics, RAS"  style="height: 100px; margin: 10px"/>
<img src="https://bgsmath.cat/wp-content/uploads/2017/09/marie_curie1-300x160.jpg" alt="Horizon 2020"  style="height: 100px; margin: 10px"/>
<img src="img/logo_imagein.png" alt="IMAGE-IN"  style="height: 100px; margin: 10px"/>
<img src="https://chemometrix.gmbh/assets/images/chemometrix-logo.png" alt="Chemometrix GmbH"  style="height: 80px; margin: 10px">
<img src="https://image-in-itn.eu/wp-content/uploads/2020/09/IPHTLogo_neu_rgb_01_de.png" alt="Leibniz-IPHT" style="height: 100px; margin: 10px">
<img src="https://image-in-itn.eu/wp-content/uploads/2020/09/bmd-1024x213.png" alt="BMD Software"  style="height: 80px; width: 250px; margin: 10px">