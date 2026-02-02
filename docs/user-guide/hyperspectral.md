# Hyperspectal images and multidimensional data

`SpectraFrame` stores spectra in an **unfolded** form: each row in `sf.spc` is one
spectrum and sample coordinates / metadata live in `sf.data` (e.g. `y`, `x`,
`batch`, `t`, `rep`, ...). For hyperspectral images and other gridded measurements
it is often convenient to work with **dense tensors** such as `(y, x, wl)` or
`(batch, y, x, wl)`.

For that, `SpectraFrame` provides [`einops`](https://einops.rocks/)-powered helpers:

- `sf.rearrange(...)` → reshape to a dense `np.ndarray`
- `sf.reduce(...)` → reduce over axes implied by the output pattern

!!! note "Optional dependency"
    These helpers require `einops`. You can install it via:

    ```bash
    pip install einops
    ```

## Conventions / assumptions

- `wl` denotes the spectral axis (`sf.wl` / columns of `sf.spc`) and is treated as the
  last axis in patterns.
- Any other named axes (e.g., `batch`, `y`, `x`, `t`, `rep`, `z`, `ch`) refer to
  columns in `sf.data`.
- Before the rearrangement/reduction takes place, the rows are sorted by the named axes
  in the pattern to make results independent of original row order.
- The *input* to einops is conceptually `(col1 col2 ... colN) wl`, therefore the provided pattern
  represents only the right side of the einops pattern.
- When `reduce` applied, the reduction is happening by a fabricated dimension (`rest`)
  which corresponds to unique combination of the ommited columns

## Setup
In the examples below we assume:

```python
import numpy as np
import pandas as pd
import pyspc
```

You can build a minimal hyperspectral image `SpectraFrame` like this:

```python
wl = np.linspace(600, 700, 4)
rows = []
spc_rows = []

for y in range(2):
    for x in range(3):
        rows.append({"y": y, "x": x})
        spc_rows.append((10 * y + x) + np.arange(len(wl), dtype=float))

sf = pyspc.SpectraFrame(np.asarray(spc_rows), wl=wl, data=pd.DataFrame(rows))
```

## Rearranging to dense tensors (`sf.rearrange`)

`sf.rearrange(pattern, ...)` takes an **einops-style output pattern** and returns
a dense `np.ndarray`. In contrast to raw `einops.rearrange`, you provide **only
the right-hand side** of the pattern — the input side is inferred from
`sf.data` + `wl`.

- `wl` refers to the spectral axis (`sf.wl`) and must be present in `pattern`.
- Any other named axes refer to columns in `sf.data`.
- Parentheses work as in `einops` (grouping / flattening axes).
- `...` (ellipsis) is not supported.

```python
# Hyperspectral cube: (y, x, wl)
cube = sf.rearrange("y x wl")

# Multiple images: (batch, y, x, wl)
cube = sf.rearrange("batch y x wl")

# Stack images vertically: ((batch*y), x, wl)
stacked = sf.rearrange("(batch y) x wl")

# Flatten pixels: (batch, (y*x), wl)
pixels = sf.rearrange("batch (y x) wl")

# Add a singleton axis (e.g. a "channel" dim): (batch, y, x, 1, wl)
cube_ch = sf.rearrange("batch y x 1 wl")
```

### Ragged grids and padding (`fill_value` and `grid_values`)

If some coordinate combinations are missing (a **ragged** grid), `sf.rearrange(...)`
automatically pads the missing entries. By default, missing spectra are filled
with NaNs (`np.nan`), but you can override this via `fill_value=...`.

Note: If padding is applied, the output dtype may be promoted to accommodate `fill_value` (e.g. integer spectra padded with `np.nan` become floats).

```python
# Fill missing pixels with NaNs (default behavior)
cube = sf.rearrange("y x wl")

# Fill missing pixels with a custom value
cube0 = sf.rearrange("y x wl", fill_value=0.0)
```

You can also explicitly specify the grid for one or more axes via `grid_values`.
This is useful for padding images or forcing a specific axis ordering.

```python
# Pad x to include an extra column (x=3), filled with NaNs
cube = sf.rearrange("y x wl", fill_value=np.nan, x=[0, 1, 2, 3])
```

## Reducing along axes (`sf.reduce`)

`sf.reduce(reducer, pattern, ...)` keeps the axes named in `pattern` and reduces
over all other axes:

- Include `wl` (as the last axis) to **keep spectra**.
- Omit `wl` to **reduce over wavelengths** and return scalars / images.

`reducer` can be:

- a string: `"mean"`, `"sum"`, `"min"`, `"max"`, `"std"`, `"median"`
- a callable (e.g. `np.mean`, `np.nanmedian`, ...)

For supported string reducers, `ignore_na=True` switches to the corresponding
NaN-aware NumPy variant (e.g. `"mean"` → `np.nanmean`).

```python
# Mean intensity map (reduce over wl): ((batch*y), x)
img = sf.reduce("mean", "(batch y) x")

# Average over x but keep spectra: (batch, y, wl)
mean_y = sf.reduce("mean", "batch y wl")

# Use a callable reducer (ignores NaNs by design)
img_robust = sf.reduce(np.nanmedian, "(batch y) x")
```

### Reductions with missing combinations

Missing coordinate combinations are padded with NaNs by default. For supported
string reducers, set `ignore_na=True` to ignore these NaNs during reduction:

```python
# Example: average spectra per (y, x) even if some replicates are missing
mean_cube = sf.reduce(
    "mean",
    "y x wl",
    ignore_na=True,
)
```

## Troubleshooting

- `ValueError: Pattern must include 'wl'` → `sf.rearrange(...)` always requires `wl`.
- `ValueError: Pattern references axes not present...` → axis names must exist in `sf.data`.
- `ValueError: Duplicate coordinate combinations...` → include additional axis columns
  (e.g. a replicate ID) in the pattern, or aggregate first.
- `NotImplementedError: Ellipsis (...) ...` → `...` is currently unsupported in patterns.
