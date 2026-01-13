# Immediate Action Items

**Priority:** CRITICAL
**Owner:** Development Team
**Created:** 2026-01-13

These are the highest-priority tasks that should be completed in the next 1-2 weeks to begin making this project production-ready.

---

## Week 1: Quick Wins (4-8 hours)

### 1. Fix Configuration Issues ⚡ (30 minutes)

**File:** `codex.yaml`

Replace placeholders:
```yaml
# BEFORE (lines 6-9):
project:
  name: YOUR_REPO_NAME
  label: "YOUR HUMAN-FRIENDLY LABEL"
  language: python
  entry_package: YOUR_TOP_LEVEL_PACKAGE

# AFTER:
project:
  name: nonstationarity_toolbox
  label: "Nonstationarity Toolbox"
  language: python
  entry_package: nonstationarity_toolbox
```

---

### 2. Create Essential Documentation Files 📝 (2 hours)

#### a. Expand README.md
```markdown
# Nonstationarity Toolbox

A comprehensive Python toolkit for analyzing and modeling nonstationary time series data.

## Status

⚠️ **Alpha Stage (v0.1.0)** - Under active development. Not ready for production use.

## Features (Planned)

### Statistical Tests
- Unit root tests (ADF, KPSS, PP, ZA)
- Structural break detection
- Volatility change tests
- Long memory tests (Hurst exponent, GPH)
- Nonlinearity tests (BDS, McLeod-Li)

### Time Series Models
- ARIMA/SARIMA models
- GARCH and Stochastic Volatility models
- Markov-Switching models
- Time-Varying Parameter models
- Long memory (ARFIMA) models

### Workflows
- Automated diagnostic pipelines
- Model selection and comparison
- Visualization and reporting

### Interfaces
- Command-line interface (CLI)
- Interactive web dashboard (Streamlit)

## Installation (Development)

```bash
git clone https://github.com/cwester75/nonstationarity_toolbox.git
cd nonstationarity_toolbox
pip install -e .[dev]
```

## Quick Start

⚠️ Core functionality not yet implemented. Coming soon!

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
pytest

# Or use the test orchestrator
python scripts/codex_runner.py --combo standard
```

### Code Quality

```bash
# Format code
black .

# Type checking
mypy nonstationarity_toolbox

# Linting
flake8 nonstationarity_toolbox
```

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in academic research, please cite:

```bibtex
@software{nonstationarity_toolbox,
  title = {Nonstationarity Toolbox},
  author = {Nonstationarity Toolbox Contributors},
  year = {2026},
  url = {https://github.com/cwester75/nonstationarity_toolbox}
}
```

## Roadmap

- [ ] Phase 1: Core statistical tests (Q1 2026)
- [ ] Phase 2: Time series models (Q2 2026)
- [ ] Phase 3: Workflows and interfaces (Q3 2026)
- [ ] Phase 4: Production release v1.0 (Q4 2026)

## Links

- Documentation: (Coming soon)
- Issues: https://github.com/cwester75/nonstationarity_toolbox/issues
- Repository: https://github.com/cwester75/nonstationarity_toolbox
```

#### b. Create LICENSE file
```text
MIT License

Copyright (c) 2026 Nonstationarity Toolbox Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#### c. Create CONTRIBUTING.md
```markdown
# Contributing to Nonstationarity Toolbox

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install in development mode:
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests and quality checks:
   ```bash
   # Format code
   black .

   # Type checking
   mypy nonstationarity_toolbox

   # Linting
   flake8 nonstationarity_toolbox

   # Run tests
   pytest
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. Push and create a pull request

## Code Quality Standards

- **Code formatting:** Use Black (100 char line length)
- **Type hints:** Add type annotations to all functions
- **Docstrings:** Use NumPy-style docstrings
- **Testing:** Write tests for all new features (aim for >80% coverage)
- **Linting:** Code must pass flake8 checks

## Testing Guidelines

- Place tests in the appropriate tier:
  - `tests/unit/` - Fast, isolated unit tests
  - `tests/integration/` - Module integration tests
  - `tests/scenario/` - End-to-end tests with real data
  - `tests/stress/` - Performance and load tests

- Use pytest markers:
  ```python
  @pytest.mark.unit
  def test_function():
      assert function() == expected
  ```

## Pull Request Process

1. Update documentation if you change behavior
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Coding Conventions

### Python Style
- Follow PEP 8
- Use meaningful variable names
- Keep functions focused and small
- Avoid deep nesting

### Statistical Code
- Include references to papers/algorithms
- Add mathematical notation in docstrings (LaTeX)
- Validate numerical stability
- Handle edge cases (NaN, Inf, etc.)

## Questions?

Open an issue or start a discussion on GitHub.
```

#### d. Create CHANGELOG.md
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Test orchestration framework (codex_runner.py)
- Configuration files (pyproject.toml, codex.yaml)
- Production readiness review documentation

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [0.1.0] - 2026-01-13

### Added
- Initial alpha release
- Project scaffolding and architecture
- Module structure for diagnostics, models, workflows, interfaces, and utils

[Unreleased]: https://github.com/cwester75/nonstationarity_toolbox/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cwester75/nonstationarity_toolbox/releases/tag/v0.1.0
```

---

### 3. Create Test Directory Structure 🧪 (15 minutes)

```bash
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/scenario
mkdir -p tests/stress
touch tests/__init__.py
touch tests/conftest.py
```

**File:** `tests/conftest.py`
```python
"""Shared pytest fixtures for all tests."""
import pytest
import numpy as np


@pytest.fixture
def sample_timeseries():
    """Generate a simple time series for testing."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def stationary_series():
    """Generate a stationary time series."""
    np.random.seed(42)
    return np.random.randn(500)


@pytest.fixture
def nonstationary_series():
    """Generate a nonstationary time series (random walk)."""
    np.random.seed(42)
    innovations = np.random.randn(500)
    return np.cumsum(innovations)
```

**File:** `tests/unit/test_placeholder.py`
```python
"""Placeholder test to verify test infrastructure works."""
import pytest


@pytest.mark.unit
def test_basic_math():
    """Basic test to ensure pytest works."""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_imports():
    """Verify package can be imported."""
    import nonstationarity_toolbox
    assert nonstationarity_toolbox.__version__ == "0.1.0"


@pytest.mark.unit
def test_sample_fixture(sample_timeseries):
    """Test that fixtures work."""
    assert len(sample_timeseries) == 100
    assert sample_timeseries.shape == (100,)
```

---

### 4. Update Dependencies 📦 (30 minutes)

**File:** `pyproject.toml` (update lines 28-42)

```toml
dependencies = [
    "numpy>=2.0.0,<3.0.0",
    "pandas>=2.2.0,<3.0.0",
    "scipy>=1.14.0,<2.0.0",
    "matplotlib>=3.9.0,<4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "flake8>=7.0.0",
    "mypy>=1.11.0",
    "hypothesis>=6.100.0",  # Add property-based testing
]
streamlit = [
    "streamlit>=1.38.0",
]
```

---

### 5. Set Up Pre-commit Hooks ⚙️ (30 minutes)

**File:** `.pre-commit-config.yaml` (create new)
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
```

Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

---

### 6. Set Up GitHub Actions 🚀 (1 hour)

**File:** `.github/workflows/test.yml` (create new)
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests with pytest
      run: |
        pytest --cov=nonstationarity_toolbox --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

**File:** `.github/workflows/lint.yml` (create new)
```yaml
name: Lint

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Check formatting with Black
      run: black --check .

    - name: Lint with flake8
      run: flake8 nonstationarity_toolbox --max-line-length=100 --extend-ignore=E203,W503

    - name: Type check with mypy
      run: mypy nonstationarity_toolbox --ignore-missing-imports
```

---

## Week 2: First Implementation (8-16 hours)

### 7. Create Exception Hierarchy 🔥 (1 hour)

**File:** `nonstationarity_toolbox/exceptions.py` (create new)
```python
"""Custom exceptions for nonstationarity_toolbox."""


class NonstationarityToolboxError(Exception):
    """Base exception for all nonstationarity_toolbox errors."""
    pass


class DataValidationError(NonstationarityToolboxError):
    """Raised when input data validation fails."""
    pass


class InsufficientDataError(DataValidationError):
    """Raised when insufficient data points are provided."""
    pass


class InvalidParameterError(NonstationarityToolboxError):
    """Raised when invalid parameters are provided."""
    pass


class ConvergenceError(NonstationarityToolboxError):
    """Raised when iterative algorithms fail to converge."""
    pass


class ModelNotFittedError(NonstationarityToolboxError):
    """Raised when attempting to use a model that hasn't been fitted."""
    pass


class NumericalInstabilityError(NonstationarityToolboxError):
    """Raised when numerical computations become unstable."""
    pass
```

---

### 8. Implement Logging Infrastructure 📊 (1 hour)

**File:** `nonstationarity_toolbox/logging_config.py` (create new)
```python
"""Logging configuration for nonstationarity_toolbox."""
import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up logging for nonstationarity_toolbox.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Path, optional
        Path to log file. If None, log to console only.
    format_string : str, optional
        Custom format string. If None, use default.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    logger = logging.getLogger("nonstationarity_toolbox")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(f"nonstationarity_toolbox.{name}")
```

Update `__init__.py` to include logging setup:
```python
"""Nonstationarity Toolbox.

A comprehensive toolkit for analyzing and modeling nonstationary time series data.
"""
from nonstationarity_toolbox.logging_config import setup_logging, get_logger

__version__ = "0.1.0"
__all__ = ["setup_logging", "get_logger", "__version__"]
```

---

### 9. Implement Data Validation Utilities 🛡️ (2-3 hours)

**File:** `nonstationarity_toolbox/utils/data_utils.py`
```python
"""Data validation and preprocessing utilities."""
import logging
from typing import Union

import numpy as np
import pandas as pd

from nonstationarity_toolbox.exceptions import (
    DataValidationError,
    InsufficientDataError,
)
from nonstationarity_toolbox.logging_config import get_logger

logger = get_logger(__name__)


def validate_timeseries(
    data: Union[np.ndarray, pd.Series],
    min_length: int = 10,
    allow_nan: bool = False,
) -> np.ndarray:
    """
    Validate time series data.

    Parameters
    ----------
    data : array-like
        Time series data to validate
    min_length : int, default=10
        Minimum required length
    allow_nan : bool, default=False
        Whether to allow NaN values

    Returns
    -------
    np.ndarray
        Validated and converted time series as 1D numpy array

    Raises
    ------
    DataValidationError
        If data is invalid
    InsufficientDataError
        If data length is insufficient
    """
    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.values
    elif isinstance(data, (list, tuple)):
        arr = np.array(data)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise DataValidationError(
            f"Unsupported data type: {type(data)}. "
            "Expected numpy array, pandas Series, list, or tuple."
        )

    # Ensure 1D
    if arr.ndim != 1:
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.flatten()
        else:
            raise DataValidationError(
                f"Expected 1D time series, got shape {arr.shape}"
            )

    # Check length
    if len(arr) < min_length:
        raise InsufficientDataError(
            f"Insufficient data: got {len(arr)} points, "
            f"need at least {min_length}"
        )

    # Check for NaN
    if not allow_nan and np.any(np.isnan(arr)):
        raise DataValidationError(
            "Time series contains NaN values. "
            "Set allow_nan=True to permit NaN values."
        )

    # Check for Inf
    if np.any(np.isinf(arr)):
        raise DataValidationError("Time series contains infinite values")

    logger.debug(f"Validated time series with {len(arr)} observations")
    return arr


def remove_trend(
    data: np.ndarray, method: str = "linear"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove trend from time series.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    method : str, default="linear"
        Detrending method: 'linear', 'constant', or 'none'

    Returns
    -------
    detrended : np.ndarray
        Detrended time series
    trend : np.ndarray
        Estimated trend component

    Raises
    ------
    ValueError
        If method is invalid
    """
    if method == "none":
        return data, np.zeros_like(data)

    if method == "constant":
        trend = np.full_like(data, np.mean(data))
        return data - trend, trend

    if method == "linear":
        t = np.arange(len(data))
        coeffs = np.polyfit(t, data, deg=1)
        trend = np.polyval(coeffs, t)
        return data - trend, trend

    raise ValueError(
        f"Invalid detrending method: {method}. "
        "Choose from: 'linear', 'constant', 'none'"
    )


def check_stationarity_visual(data: np.ndarray, window: int = 50) -> dict:
    """
    Quick visual stationarity check using rolling statistics.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    window : int, default=50
        Window size for rolling statistics

    Returns
    -------
    dict
        Dictionary with rolling mean and std statistics
    """
    if len(data) < window:
        raise InsufficientDataError(
            f"Data length ({len(data)}) must be >= window ({window})"
        )

    # Convert to pandas for rolling calculations
    series = pd.Series(data)
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    return {
        "rolling_mean": rolling_mean.values,
        "rolling_std": rolling_std.values,
        "mean_of_means": np.nanmean(rolling_mean),
        "std_of_stds": np.nanstd(rolling_std),
    }
```

---

### 10. Create First Real Test 🧪 (1 hour)

**File:** `tests/unit/test_data_utils.py`
```python
"""Tests for data validation utilities."""
import numpy as np
import pandas as pd
import pytest

from nonstationarity_toolbox.utils.data_utils import (
    validate_timeseries,
    remove_trend,
    check_stationarity_visual,
)
from nonstationarity_toolbox.exceptions import (
    DataValidationError,
    InsufficientDataError,
)


@pytest.mark.unit
class TestValidateTimeseries:
    """Tests for validate_timeseries function."""

    def test_numpy_array(self):
        """Test with numpy array input."""
        data = np.random.randn(100)
        result = validate_timeseries(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100

    def test_pandas_series(self):
        """Test with pandas Series input."""
        data = pd.Series(np.random.randn(100))
        result = validate_timeseries(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100

    def test_list_input(self):
        """Test with list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        result = validate_timeseries(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InsufficientDataError):
            validate_timeseries(data, min_length=10)

    def test_nan_values_rejected(self):
        """Test that NaN values are rejected by default."""
        data = np.array([1.0, 2.0, np.nan, 4.0] * 25)
        with pytest.raises(DataValidationError, match="NaN"):
            validate_timeseries(data)

    def test_nan_values_allowed(self):
        """Test that NaN values can be allowed."""
        data = np.array([1.0, 2.0, np.nan, 4.0] * 25)
        result = validate_timeseries(data, allow_nan=True)
        assert len(result) == 100

    def test_inf_values_rejected(self):
        """Test that Inf values are rejected."""
        data = np.array([1.0, 2.0, np.inf, 4.0] * 25)
        with pytest.raises(DataValidationError, match="infinite"):
            validate_timeseries(data)

    def test_2d_array_single_column(self):
        """Test that 2D array with single column is flattened."""
        data = np.random.randn(100, 1)
        result = validate_timeseries(data)
        assert result.ndim == 1
        assert len(result) == 100

    def test_invalid_shape(self):
        """Test that multi-column 2D array is rejected."""
        data = np.random.randn(100, 2)
        with pytest.raises(DataValidationError, match="1D"):
            validate_timeseries(data)


@pytest.mark.unit
class TestRemoveTrend:
    """Tests for remove_trend function."""

    def test_linear_detrend(self):
        """Test linear detrending."""
        # Create data with linear trend
        t = np.arange(100)
        trend = 2 * t + 5
        noise = np.random.randn(100) * 0.1
        data = trend + noise

        detrended, estimated_trend = remove_trend(data, method="linear")

        # Check that trend is removed (mean should be close to 0)
        assert abs(np.mean(detrended)) < 1.0
        # Check that estimated trend is close to true trend
        assert np.allclose(estimated_trend, trend, atol=0.5)

    def test_constant_detrend(self):
        """Test constant detrending (removing mean)."""
        data = np.random.randn(100) + 10.0  # Mean of 10
        detrended, trend = remove_trend(data, method="constant")

        assert abs(np.mean(detrended)) < 0.1
        assert np.allclose(trend, 10.0, atol=0.5)

    def test_no_detrend(self):
        """Test no detrending."""
        data = np.random.randn(100)
        detrended, trend = remove_trend(data, method="none")

        assert np.array_equal(detrended, data)
        assert np.array_equal(trend, np.zeros_like(data))

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="Invalid detrending method"):
            remove_trend(data, method="invalid")


@pytest.mark.unit
class TestCheckStationarityVisual:
    """Tests for check_stationarity_visual function."""

    def test_stationary_series(self, stationary_series):
        """Test with stationary series."""
        result = check_stationarity_visual(stationary_series, window=50)

        assert "rolling_mean" in result
        assert "rolling_std" in result
        assert "mean_of_means" in result
        assert "std_of_stds" in result

        # For stationary series, std of rolling means should be small
        assert result["std_of_stds"] < 1.0

    def test_nonstationary_series(self, nonstationary_series):
        """Test with nonstationary series (random walk)."""
        result = check_stationarity_visual(nonstationary_series, window=50)

        # Random walk should have increasing volatility
        assert result["std_of_stds"] > 0.5

    def test_insufficient_data_for_window(self):
        """Test with data shorter than window."""
        data = np.random.randn(30)
        with pytest.raises(InsufficientDataError):
            check_stationarity_visual(data, window=50)
```

---

## Verification Steps

After completing the above tasks, verify:

1. **Configuration fixed:**
   ```bash
   grep -A4 "^project:" codex.yaml  # Should show real values, not placeholders
   ```

2. **Tests work:**
   ```bash
   pytest -v  # Should pass with >3 tests
   python scripts/codex_runner.py --combo smoke  # Should run unit tests
   ```

3. **Linting works:**
   ```bash
   black --check .
   flake8 nonstationarity_toolbox
   mypy nonstationarity_toolbox
   ```

4. **Pre-commit works:**
   ```bash
   pre-commit run --all-files
   ```

5. **CI works:**
   - Push to GitHub
   - Check GitHub Actions tab
   - Both test.yml and lint.yml should pass

---

## Success Criteria

Week 1-2 is successful when:

- ✅ All configuration placeholders are fixed
- ✅ README is comprehensive and informative
- ✅ Essential files created (LICENSE, CONTRIBUTING, CHANGELOG)
- ✅ Test infrastructure working (pytest passes)
- ✅ At least one module fully implemented with tests
- ✅ Dependencies updated to current versions
- ✅ CI/CD pipeline passing
- ✅ Pre-commit hooks installed and working
- ✅ Code coverage >80% for implemented modules

---

**Estimated Total Time:** 12-16 hours
**Expected Completion:** 1-2 weeks (part-time) or 2-3 days (full-time)
**Next Steps:** See PRODUCTION_READINESS_CHECKLIST.md Phase 2
