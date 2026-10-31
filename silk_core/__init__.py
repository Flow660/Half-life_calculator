"""
SILK half-life core package.

A Python re-implementation of the scientific core of the MATLAB SILK
(Stable-Isotope-Labelling Kinetics) toolbox: it reads pulse / pulse-and-chase
proteomics labelling data, fits a precursor (Hsol) labelling curve and derives a
turnover half-life for every protein of interest (Hpoi).

Modules
-------
loaders    : read the four SILK Excel formats into tidy numpy/pandas structures
models     : the two kinetic prediction models (single-exponential & 2-compartment)
precursor  : fit the soluble precursor (Hsol) bi-exponential curve
fitting    : fit the per-protein turnover parameter and convert it to a half-life
pipeline   : glue the above into an end-to-end run
quality    : per-protein fit-quality scores + goodness-of-fit filter (+ bootstrap CI)
simulation : noise-driven recovery simulation to justify the reliable half-life range
"""

from . import loaders, models, precursor, fitting, pipeline, quality, simulation  # noqa: F401

__all__ = ["loaders", "models", "precursor", "fitting", "pipeline", "quality", "simulation"]
__version__ = "1.1.0"
