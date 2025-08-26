# HCLIM Utilities

Welcome to the documentation for **HCLIM Utilities**, a collection of Python functions developed during the 2025 internship at **Met Éireann**.  

These utilities provide convenient building blocks for analysing and visualising 
climate model output, with a focus on the HCLIM regional climate model and related datasets.

---

## Overview

This package contains tools for:

- **Data I/O**  
  Functions to load GRIB and NetCDF files into [`iris`](https://scitools-iris.readthedocs.io/) 
  cubes and ensure consistent attributes and time handling.

- **Gridding and regridding**  
  Utilities to create target rectilinear grids, regrid cubes, and extract common 
  periods across multiple datasets.

- **Diagnostics and statistics**  
  Functions to compute rainfall accumulations, biases, diurnal cycles, and add 
  summary statistics (MAE, bias, standard deviation) directly to plots.

- **Plotting**  
  Ready-made plotting functions for:
  - Spatial parameter maps (temperature, pressure, precipitation, etc.)
  - Bias plots and animations
  - Cyclone tracks
  - Time series at specific locations or averaged over a region

- **Animations**  
  Functions for producing animated weather maps or parameter fields over time, 
  useful for case study analysis and communication.

---

## Getting Started

### Requirements

- Python 3.11 (recommended via Conda)
- [Iris](https://scitools-iris.readthedocs.io/)  
- [Cartopy](https://scitools.org.uk/cartopy/docs/latest/)  
- [Xarray](https://docs.xarray.dev/)  
- [Matplotlib](https://matplotlib.org/)  
- [Tqdm](https://tqdm.github.io/)  
