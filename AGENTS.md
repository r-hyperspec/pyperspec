# Project overview

This project is a Python package for spectral data analysis, providing tools for handling, processing, and visualizing spectral datasets.
It is inspired by R's `hyperSpec` package but tailored for Python ecosystem, i.e. close to libraries such as `numpy`, `pandas`, and `scikit-learn`.
Central concept is `SpectraFrame`, a data structure combining three components: spectral data (`spc` - 2D numpy array), metadata (`data` - pandas DataFrame), and spectral coordinates (`wl` - 1D numpy array).
The package does not aim to re-implement all algorithms, but rather to provide a convenient interface that orchestrates between existing libraries and methods, e.g. `SpectraFrame.apply` method dispatches to the underlying spectral data component.

# Development guidelines

## Setup instructions:
* Python 3.9+
* Use `poetry` for management
* Update the package version in `pyproject.toml`, if not updated yet
* If available, use virtual environment specified in '.vscode/settings.json -> python.defaultInterpreterPath` for running python commands
* Assume dev environment has all packages installed

## Development workflow:
* Create a new branch for each feature/bugfix
* Follow TDD principles: write tests before implementing features. Make sure to cover edge cases.
* Don't run tests and linter unless explicitly asked.
* If asked to, use `poetry run pytest --doctest-modules` to run tests.
* **Test interface behavior, not implementation details:** Tests should verify
that the public API behaves correctly, not how it's implemented internally. This
makes tests resilient to refactoring and ensures the contract with users remains
intact.
* Do not run tests or linters unless explicitly requested.
* Do not preserve backward compatibility unless explicitly requested.

## Code style guidelines:
* Follow PEP8 guidelines
* Use type hints throughout the codebase
* For linting use black and ruff:
  - `poetry run black .`
  - `poetry run ruff check --fix .`
* Use numpydoc style for docstrings. For public functions/methods, include examples helpful for users.
* Each block of code should have a *short* explaining comment.

## Documentation guidelines:
Check and update documentation in `docs/` folder as needed.
Specific classes and modules go into separate files with `api-` prefix, like `api-peaks.md`, `api-spectraframe.md`, etc.
Practical usage examples go into `user-guide` folder.

## Python tips

### General Python Best Practices

*   **Constants:** Use immutable global constant collections (tuple, frozenset, immutabledict) to avoid hard-to-find bugs. Prefer constants over wild string/int literals, especially for dictionary keys, pathnames, and enums.
*   **Naming:** Name mappings like `value_by_key` to enhance readability in lookups (e.g., `item = item_by_id[id]`).
*   **Readability:** Use f-strings for concise string formatting, but use lazy-evaluated `%`-based templates for logging. Use `repr()` or `pprint.pformat()` for human-readable debug messages. Use `_` as a separator in numeric literals to improve readability.
*   **Comprehensions:** Use list, set, and dict comprehensions for building collections concisely.
*   **Iteration:** Iterate directly over containers without indices. Use `enumerate()` when you need the index, `dict.items()` for keys and values, and `zip()` for parallel iteration.
*   **Built-ins:** Leverage built-in functions like `all()`, `any()`, `reversed()`, `sum()`, etc., to write more concise and efficient code.
*   **Flattening Lists:** Use `itertools.chain.from_iterable()` to flatten a list of lists efficiently without unnecessary copying.
*   **String Methods:** Use `startswith()` and `endswith()` with a tuple of strings to check for multiple prefixes or suffixes at once.
*   **Decorators:** Use decorators to add common functionality (like logging, timing, caching) to functions without modifying their core logic. Use `functools.wraps()` to preserve the original function's metadata.
*   **Context Managers:** Use `with` statements and context managers (from `contextlib` or custom classes with `__enter__`/`__exit__`) to ensure resources are properly initialized and torn down, even in the presence of exceptions.
*   **Else Clauses:** Utilize the `else` clause in `try/except` blocks (runs if no exception), and in `for/while` loops (runs if the loop completes without a `break`) to write more expressive and less error-prone code.
*   **Single Assignment:** Prefer single-assignment form (assign to a variable once) over assign-and-mutate to reduce bugs and improve readability. Use conditional expressions where appropriate.
*   **Equality vs. Identity:** Use `is` or `is not` for singleton comparisons (e.g., `None`, `True`, `False`). Use `==` for value comparison.
*   **Default Arguments:** NEVER use mutable default arguments. Use `None` as a sentinel value instead.
*   **`__add__()` vs. `__iadd__()`:** `x += y` (in-place add) can modify the object in-place if `__iadd__` is implemented (like for lists), while `x = x + y` creates a new object. This matters when multiple variables reference the same object.
*   **Properties:** Use `@property` to create getters and setters only when needed, maintaining a simple attribute access syntax. Avoid properties for computationally expensive operations or those that can fail.
*   **Modules for Namespacing:** Use modules as the primary mechanism for grouping and namespacing code elements, not classes. Avoid `@staticmethod` and methods that don't use `self`.
*   **Argument Passing:** Python is call-by-value, where the values are object references (pointers). Assignment binds a name to an object. Modifying a mutable object through one name affects all names bound to it.
*   **Keyword/Positional Arguments:** Use `*` to force keyword-only arguments and `/` to force positional-only arguments. This can prevent argument transposition errors and make APIs clearer, especially for functions with multiple arguments of the same type.
*   **Type Hinting:** Annotate code with types to improve readability, debuggability, and maintainability. Use abstract types from `collections.abc` for container annotations (e.g., `Sequence`, `Mapping`, `Iterable`). Annotate return values, including `None`. Choose the most appropriate abstract type for function arguments and return types.
*   **`NewType`:** Use `typing.NewType` to create distinct types from primitives (like `int` or `str`) to prevent argument transposition and improve type safety.
*   **`__repr__()` vs. `__str__()`:** Implement `__repr__()` for unambiguous, developer-focused string representations, ideally evaluable. Implement `__str__()` for human-readable output. `__str__()` defaults to `__repr__()`.
*   **F-string Debug:** Use `f"{expr=}"` for concise debug printing, showing both the expression and its value.

### Testing

*   **Assertions:** Use pytest's native `assert` statements with informative expressions. Pytest automatically provides detailed failure messages showing the values involved. Add custom messages with `assert condition, "helpful message"` when the expression alone isn't clear.
*   **Custom Assertions:** Write reusable helper functions (not methods) for repeated complex checks. Use `pytest.fail("message")` to explicitly fail a test with a custom message.
*   **Parameterized Tests:** Use `@pytest.mark.parametrize` to reduce duplication when running the same test logic with different inputs.
*   **Fixtures:** Use pytest fixtures (with `@pytest.fixture`) for test setup, teardown, and dependency injection. Fixtures are cleaner than class-based setup methods and can be easily shared across tests.
*   **Mocking:** Use `mock.create_autospec()` with `spec_set=True` to create mocks that match the original object's interface, preventing typos and API mismatch issues. Use context managers (`with mock.patch(...)`) to manage mock lifecycles and ensure patches are stopped. Prefer injecting dependencies via fixtures over patching.
*   **Asserting Mock Calls:** Use `mock.ANY` and other matchers for partial argument matching when asserting mock calls (e.g., `assert_called_once_with`).
*   **Temporary Files:** Use pytest's `tmp_path` and `tmp_path_factory` fixtures for creating isolated and automatically cleaned-up temporary files/directories. These are preferred over the `tempfile` module in pytest tests.
*   **Avoid Randomness:** Make sure to set random seeds in tests involving randomness to ensure reproducibility. Avoid flaky tests that pass/fail nondeterministically.
*   **Test Invariants:** Focus tests on the invariant behaviors of public APIs, not implementation details.
*   **Test Organization:** Prefer simple test functions over class-based tests unless you need to share fixtures across multiple test methods in a class. Use descriptive test names that explain the behavior being tested.