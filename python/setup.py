"""
setup.py for the llmquant Python extension.

This script builds the C++20 LLMTokenStreamQuantEngine Python bindings
using pybind11 and setuptools.

## Prerequisites

- Python 3.9+
- pybind11 >= 2.11  (`pip install pybind11`)
- A C++20-capable compiler:
    Windows: MSVC 2022 (cl.exe) or clang-cl
    Linux:   GCC 12+ or Clang 15+
    macOS:   Apple Clang 14+
- The LLMTokenStreamQuantEngine project built first via CMake (produces
  the static/shared library that setup.py links against).

## Build & install

```bash
# From this directory (LLMTokenStreamQuantEngine/python/):
pip install pybind11
python setup.py build_ext --inplace

# Or install into the current Python environment:
pip install .
```

## CMake build (recommended)

Enable the Python binding target in CMake:

```bash
cmake -B build -DLLMQUANT_ENABLE_PYTHON_BINDINGS=ON
cmake --build build --config Release
```

The resulting `llmquant.*.pyd` (Windows) or `llmquant.*.so` (Linux/macOS)
will be placed in `build/python/`.
"""

import os
import sys
from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11
except ImportError:
    raise RuntimeError(
        "pybind11 is required. Install with: pip install pybind11"
    ) from None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
SRC_DIR = PROJECT_ROOT / "src"
INCLUDE_DIR = PROJECT_ROOT / "include"

# The CMake build directory for the compiled static library.
# Adjust BUILD_DIR if you use a different build directory.
BUILD_DIR = PROJECT_ROOT / "build"

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/std:c++20", "/O2", "/EHsc", "/W3"]
else:
    extra_compile_args = ["-std=c++20", "-O2", "-Wall", "-fvisibility=hidden"]
    extra_link_args = ["-Wl,-rpath,$ORIGIN"]

# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------
# Source files needed to build the Python extension standalone.
# In a full CMake-integrated build these come from the static library;
# here we list the minimal set needed for the Python binding.
sources = [
    str(HERE / "bindings.cpp"),
    str(SRC_DIR / "LLMAdapter.cpp"),
    str(SRC_DIR / "TradeSignalEngine.cpp"),
]

ext = Extension(
    name="llmquant",
    sources=sources,
    include_dirs=[
        str(INCLUDE_DIR),
        pybind11.get_include(),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="llmquant",
    version="1.3.0",
    description=(
        "Python bindings for LLMTokenStreamQuantEngine: "
        "maps LLM token streams to quantitative trading signals"
    ),
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="LLMTokenStreamQuantEngine contributors",
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "pybind11>=2.11",
    ],
    ext_modules=[ext],
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Office/Business :: Financial",
    ],
)
