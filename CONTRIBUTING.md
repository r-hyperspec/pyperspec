# Helpful commands

```bash
# Update the documentation on GitHub Pages
mkdocs gh-deploy --clean
# Clean local mkdocs build artifacts afterwards
rm -rf site
```

```bash
# Run mkdocs locally
mkdocs serve
```

```bash
# Run tests with doctests
pytest --doctest-modules .
```