# Helpful commands

```bash
# Update the documentation on GitHub Pages
uv run mkdocs gh-deploy --clean
# Clean local mkdocs build artifacts afterwards
rm -rf site
```

```bash
# Run mkdocs locally
uv run mkdocs serve
```

```bash
# Run tests with doctests
uv run pytest --doctest-modules .
```