# Installation

## Requirements

`pyperspec` requires Python 3.9 or higher and depends on the following packages:

- `pandas >= 2.1.1`
- `matplotlib >= 3.8.0`
- `scipy >= 1.11.2`
- `pybaselines >= 1.1.0`
- `scikit-learn >= 1.6.1`

## Install from GitHub

Currently, `pyperspec` is only available from GitHub. You can install it using pip:

```bash
pip install git+https://github.com/r-hyperspec/pyperspec.git
```

To install a specific version, you can specify the commit hash or tag:

```bash
pip install git+https://github.com/r-hyperspec/pyperspec.git@<commit_hash_or_tag>
```

## Optional extras

Some features require optional dependencies:

- Multidimensional tensor reshaping and reduction (`SpectraFrame.rearrange`, `SpectraFrame.reduce`):
  install with `pip install einops`

## Verify Installation

To verify that `pyperspec` was installed correctly:

```python
import pyspc
print(pyspc.__version__)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
2. **Version Conflicts**: Consider using a virtual environment
3. **Permission Issues**: Use `--user` flag with pip install

### Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/r-hyperspec/pyperspec/issues)
2. Create a new issue with your system information and error message
