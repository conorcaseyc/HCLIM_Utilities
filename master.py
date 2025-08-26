# Import dependencies.
import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import iris
import iris.plot as iplt
from iris.util import unify_time_units, equalise_attributes
from iris.coord_categorisation import add_categorised_coord
from esmf_regrid.schemes import ESMFBilinear
import cf_units
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from pyproj import Transformer
from tqdm import tqdm, trange
from geopy.geocoders import Nominatim

def load_grib_files(folder, param, standard_name, constraint=None):
	"""
    Load multiple GRIB files from a folder, extract a parameter, and merge into a single Iris cube.

    Each GRIB file is opened with Xarray (via cfgrib), converted to an Iris cube, 
    and combined into a merged cube with unified time units and equalised attributes. 
    Optionally, a constraint can be applied to extract a subset of the data.

    Parameters
    ----------
    folder : str
        Path to the folder containing `.grib` or `.GRIB` files.
    param : str
        Variable/parameter name in the GRIB files to load (e.g., 't2m' for 2m temperature).
    standard_name : str
        Standard name to assign to the variable for consistency with CF conventions.
    constraint : iris.Constraint, optional
        Constraint to filter the merged cube (e.g., by time, level, or region). 
        Default is None.

    Returns
    -------
    processed_cube : iris.cube.Cube
        Merged Iris cube containing the requested parameter from all GRIB files.
    """
	# Get list of files.
	files = [
		f for f in os.listdir(folder)
		if os.path.isfile(os.path.join(folder, f)) and (f.endswith(".grib") or f.endswith(".GRIB"))
	]

	# Loop through files.
	ds_cubes = iris.cube.CubeList([])
	for file in files:
		# Load file using Xarray via cfgrib.
		ds_xarray = xr.open_dataset(
			"{}/{}".format(folder, file), engine="cfgrib", decode_timedelta=True
		)[param]
		ds_xarray.attrs["standard_name"] = standard_name

		# Convert to Iris cube.
		ds_cube = ds_xarray.to_iris()

		# Append to cube list.
		ds_cubes.append(ds_cube)

	# Unify time units.
	unify_time_units(ds_cubes)
	# Equalise attributes.
	equalise_attributes(ds_cubes)
	# Concatenate.
	processed_cube = ds_cubes.merge_cube()

	# Add constraint if relevant.
	if constraint != None:
		# Extract.
		processed_cube = processed_cube.extract(constraint)

	return processed_cube

def load_hclim_netcdf(files, param, constraint=None):
	"""
    Load a list of HCLIM NetCDF files, extract monthly subsets, and merge into a single cube.

    Each input file is assumed to contain data for (at most) one full month. The function 
    then constrains each cube to exactly one month using the first timestamp in the file, then 
    concatenates all processed cubes into a single Iris cube. Optionally, a further constraint 
    can be applied to the final cube.

    Parameters
    ----------
    files : list of str or str
        Paths to the NetCDF files to load. Can be a single file path or a list of paths.
    param : str or iris.Constraint
        Variable/parameter to load from each file.
    constraint : iris.Constraint, optional
        Constraint to apply to the final merged cube (e.g., for a subset by time, level, or region). 
        Default is None.

    Returns
    -------
    processed_cube : iris.cube.Cube
        Concatenated Iris cube containing the monthly subsets from all NetCDF files.
    """
	# Load list of NetCDF files.
	raw_cubes = iris.load(files, param)

	# Loop through list of files.
	processed_cubes = iris.cube.CubeList([])
	for it in range(len(raw_cubes)):
		# Get cube.
		raw_cube = raw_cubes[it]

		# Make sure each file only contains one month.
		# Get time coordinate.
		cube_it_time = raw_cubes[it].coord("time")
		# Get relevant date of the first day of the month.
		first_date = cube_it_time.units.num2date(
			cube_it_time.points
		)[0]
		first_date = datetime(
			first_date.year, first_date.month, first_date.day,
			first_date.hour, first_date.minute, first_date.second
		)
		# First day of next month.
		last_date = first_date + relativedelta(months=+1)
		# Create constraint.
		month_constraint = iris.Constraint(
			time=lambda cell: first_date <= cell.point < last_date
		)
		# Extract based on constraint.
		cube = raw_cube.extract(month_constraint)

		# Append to list.
		processed_cubes.append(cube)

	# Concatenate into a single cube.
	# Unify time units.
	unify_time_units(processed_cubes)
	# Equalise attributes.
	equalise_attributes(processed_cubes)
	# Concatenate.
	processed_cube = processed_cubes.concatenate_cube()

	# Add constraint if relevant.
	if constraint != None:
		# Extract.
		processed_cube = processed_cube.extract(constraint)

	return processed_cube

def calculate_rainfall_accumulation(pr_cube_input):
	"""
    Compute daily (09Z–09Z) rainfall accumulations from a time-series precipitation cube.

    The function creates a categorised time coordinate that bins timesteps into
    daily windows defined from 09:00 UTC on day *N* to 08:59:59 UTC on day *N+1*
    (labelled internally at 21:00 UTC). It then sums precipitation over each bin,
    converts the result to millimetres, and returns a cube of daily totals.
    The first and last (potentially partial) bins are dropped.

    Parameters
    ----------
    pr_cube_input : iris.cube.Cube
        Time-series precipitation cube with a ``time`` coordinate.
        Expected units are a rate convertible to ``kg m-2 s-1`` (e.g., mm/s),
        sampled at regular **3-hourly** intervals. The 3-hourly cadence is
        assumed for the conversion using ``10800 s`` per step.

    Returns
    -------
    rainfall_cube : iris.cube.Cube
        Daily precipitation totals aggregated over 09Z–09Z windows.
        The cube has:
          - ``standard_name`` set to ``'precipitation_amount'``
          - ``long_name`` set to ``'Daily Total Precipitation Amount'``
          - ``units`` set to ``'mm'``
        The auxiliary categorisation coordinate is removed before returning.

    Notes
    -----
    - Conversion to millimetres assumes 3-hourly time steps and rate units of
      ``kg m-2 s-1``: for each step, amount = rate × 10800 s, and
      ``1 kg m-2 == 1 mm`` water equivalent.
    - The first and last bins are sliced off (``[1:-1]``) to avoid partial days.
    - The internal day labels are placed at 21:00 UTC corresponding to the
      end of each 09Z–09Z window.
    """
	# Create coordinates categorisation function for days bewteen 09Z09.
	def day_09z09(cube, coord, name="day_09z09"):
		def _get_day(coord, value):
			# Get specific timestep as datetime object.
			date = coord.units.num2date(value)

			# Define on day N at 09:00.
			ref = datetime(
				date.year, date.month, date.day, 9
			)

			# Determine which day it falls within.
			if date >= ref:
				day = datetime(date.year, date.month, date.day, 21)
			else:
				day = datetime(
					date.year, date.month, date.day, 21
				) - timedelta(days=+1) 

			return coord.units.date2num(day)

		add_categorised_coord(cube, name, coord, _get_day)

	# Add coordinates categorisation to cube.
	day_09z09(pr_cube_input, pr_cube_input.coord("time"))
	pr_cube_input.coord("day_09z09").units = pr_cube_input.coord("time").units
	
	# Calculate rainfall accumulation.
	rainfall_cube = pr_cube_input.aggregated_by("day_09z09", iris.analysis.SUM)[1:-1]

	# Convert units to mm and change standard name.
	rainfall_cube *= 10800
	rainfall_cube.standard_name = "precipitation_amount"
	rainfall_cube.long_name = "Daily Total Precipitation Amount"
	rainfall_cube.units = "mm"

	# Remove extra coordinates.
	rainfall_cube.remove_coord("day_09z09")

	return rainfall_cube

def create_rectilinear_cube(lon_min, lon_max, lat_min, lat_max, res=0.1):
	"""
    Create a dummy Iris cube on a rectilinear latitude–longitude grid.

    The function generates evenly spaced latitude and longitude points
    between given bounds at the specified resolution, then builds a 
    2D cube with zero-filled data on that grid. The cube can be used 
    as a template or target grid for regridding operations.

    Parameters
    ----------
    lon_min : float
        Minimum longitude of the grid (degrees east).
    lon_max : float
        Maximum longitude of the grid (degrees east). The range is
        half-open: the last grid point is strictly less than `lon_max`.
    lat_min : float
        Minimum latitude of the grid (degrees north).
    lat_max : float
        Maximum latitude of the grid (degrees north). The range is
        half-open: the last grid point is strictly less than `lat_max`.
    res : float, optional
        Grid spacing in degrees for both latitude and longitude.
        Default is 0.1 degrees.

    Returns
    -------
    dummy_cube : iris.cube.Cube
        A 2D cube of zeros with latitude and longitude as dimension
        coordinates. Units are degrees, with a geographic coordinate
        system (`GeogCS`) defined.

    Notes
    -----
    - The dummy data array is filled with zeros and has shape 
      `(n_lat, n_lon)`.
    - Longitude and latitude arrays are generated with 
      ``numpy.arange(start, stop, res)``, so the upper bounds
      (`lon_max`, `lat_max`) are **exclusive**.
    """
	# Define coordinates.
	lat_points = np.arange(lat_min, lat_max, res)      
	lon_points = np.arange(lon_min, lon_max, res)  

	latitude = iris.coords.DimCoord(
		lat_points, standard_name='latitude', units='degrees',
		coord_system=iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
	)
	longitude = iris.coords.DimCoord(
		lon_points, standard_name='longitude', units='degrees',
		coord_system=iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
	)

	# Create a dummy cube with the target grid.
	dummy_data = np.zeros((len(lat_points), len(lon_points)))
	dummy_cube = iris.cube.Cube(
		dummy_data,
		dim_coords_and_dims=[
			(latitude, 0),
			(longitude, 1)
		],
	)

	return dummy_cube

def regrid_cube(basis_cube, target_cube, scheme="bilinear"):
	"""
    Regrid a cube onto the grid of another cube.

    Currently only supports bilinear interpolation using the
    ESMF regridding scheme.

    Parameters
    ----------
    basis_cube : iris.cube.Cube
        The reference cube that defines the target grid 
        (lat/lon resolution, extent, coordinate system).
    target_cube : iris.cube.Cube
        The input cube to be regridded to the grid of `basis_cube`.
    scheme : {'bilinear'}, optional
        Regridding scheme to use. Currently only ``'bilinear'`` is supported,
        which applies `ESMFBilinear`. Default is ``'bilinear'``.

    Returns
    -------
    regridded_cube : iris.cube.Cube
        The regridded cube on the grid of `basis_cube`.

    Raises
    ------
    NotImplementedError
        If a scheme other than ``'bilinear'`` is specified.
    """
	# Regrid target cube to basis cube coordinates.
	if scheme == "bilinear":
		return target_cube.regrid(basis_cube, ESMFBilinear())
	else:
		raise NotImplementedError("Other regridding methods have not been implemented.")

def extract_common_period(cubes):
	"""
    Extract the common overlapping time period across multiple cubes.

    This function ensures all cubes use a standard calendar, unifies their 
    time units, determines the maximum overlapping time range, and returns 
    each cube constrained to that period.

    Parameters
    ----------
    cubes : list of iris.cube.Cube
        Input cubes with a ``time`` coordinate. Each cube may have 
        different time ranges, calendars, or units.

    Returns
    -------
    constrained_cubes : list of iris.cube.Cube
        List of cubes extracted to the common overlapping time period.
        Each cube has unified time units and standard calendar.

    Raises
    ------
    ValueError
        If no overlapping time period exists between the cubes.
    iris.exceptions.CoordinateNotFoundError
        If a cube does not have a ``time`` coordinate.
    """
	# Loop through each cube.
	common_calendar_cubes = []
	for cube in cubes:
		# Convert to standard calendar.
		cube.coord("time").units = cf_units.Unit(
			cube.coord("time").units.origin, calendar="standard"
		)
		common_calendar_cubes.append(cube)

	# Unify time units.
	unified_cubes = iris.cube.CubeList(common_calendar_cubes)
	unify_time_units(unified_cubes)

	# Extract the time coordinates for all cubes
	time_ranges = []

	# Loop through to get start and end dates.
	for cube in unified_cubes:
		# Get time coordinate.
		time_coord = cube.coord("time")

		# Get points.
		times = time_coord.units.num2date(time_coord.points)

		# Append start and end dates.
		time_ranges.append((times[0], times[-1]))
    
    # Compute the max start time and min end time
	start_times, end_times = zip(*time_ranges)
	common_start = max(start_times)
	common_end = min(end_times)

	# Raise error if there is not overlap.
	if common_start >= common_end:
		raise ValueError("No overlapping time period found across cubes.")

	# Create constraint.
	constraint = iris.Constraint(
		time=lambda cell: common_start <= cell.point <= common_end
	)
	
	# Define constrained cubes
	constrained_cubes = []
	for cube in unified_cubes:
		# Extract relevant period.
		constrained_cube = cube.extract(constraint)

		# Append to list.
		constrained_cubes.append(constrained_cube)

	return constrained_cubes

def process_datasets(ref_cube, sim_cube, param, regrid_scheme="bilinear"):
	# Convert to hPa if MSLP.
	if param == "psl":
		# Reference.
		ref_cube.convert_units("hPa")
		# Simulation cube.
		sim_cube.convert_units("hPa")
	# Calculate precipitation accumulation.
	elif param == "pr":
		sim_cube = calculate_rainfall_accumulation(sim_cube)

	# Extract common timesteps.
	sim_cube, ref_cube = extract_common_period(sim_cube, ref_cube)

	# Regrid to reference grid using bilinear interpolation.
	sim_cube_regrid = regrid_cube(ref_cube, sim_cube)

	# Calculate bias.
	bias_cube = sim_cube_regrid - ref_cube

	return ref_cube, sim_cube_regrid#, bias_cube

def add_statistical_information(cube_data, unit, ax):
	"""
    Add basic statistical information as annotations on a plot.

    Computes and displays mean absolute value, standard deviation,
    and mean bias of the input data, then adds them as anchored text
    boxes to the provided Matplotlib axis.

    Parameters
    ----------
    cube_data : array_like
        Input data array (e.g., model errors, residuals). Can be any 
        NumPy-compatible sequence of values.
    unit : str
        String label for the physical unit of the data (e.g., 'K', 'mm').
    ax : matplotlib.axes.Axes
        Matplotlib axis object to which the annotations will be added.

    Returns
    -------
    success : bool
        Returns ``True`` after successfully adding the annotations.

    Notes
    -----
    - Statistics added to the axis:
      
      * **Mean absolute value**: :math:`|\bar{x}|`  
      * **Standard deviation**: :math:`s` (with Bessel's correction, ddof=1)  
      * **Mean bias**: :math:`\bar{x}`  

    - Anchored positions:
      
      * Mean absolute value and standard deviation → bottom-right (loc=4).  
      * Mean bias → top-left (loc=2).  

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> data = np.random.randn(100)
    >>> add_statistical_information(data, 'K', ax)
    True
    """
	# Calculate relevant statistical information.
	# Calculate mean absolute error.
	txt_cube = r'$|\bar{x}| = $' + '{} {}\n'.format(
		np.round(np.mean(np.abs(cube_data)), 2), unit
	)
	# Calculate standard deviation.
	txt_cube += r'$s = $' + '{} {}'.format(
		np.round(np.std(np.abs(cube_data), ddof=1), 2), unit
	)
	# Calculate mean bias.
	txt_cube_bias = r'$\bar{x} = $' + '{} {}'.format(
		np.round(float(np.mean(cube_data)), 2), unit
	)

	# Add to plots.
	# Lower right hand side.
	ax.add_artist(AnchoredText(txt_cube, loc=4))
	# Top left hand side.
	ax.add_artist(AnchoredText(txt_cube_bias, loc=2))

	return True

def plot_dist(ds, title, subtitles, nbins, xlabel, unit, fname, size, xlim=None, ylim=None):
	"""
    Plot histograms of multiple Iris cubes in a grid layout with statistical annotations.

    Each subplot corresponds to one cube, drawn as a histogram of its flattened
    data values with the specified number of bins. Titles, labels, axis limits,
    and statistical information (mean absolute value, standard deviation, and mean bias)
    are added to each subplot. The entire figure is saved to file.

    Parameters
    ----------
    ds : list of list of iris.cube.Cube
        Nested list (2D structure) of Iris cubes. Each element `ds[i][j]` is an
        Iris cube; its `.data` values are flattened into a histogram.
    title : str
        Main title for the entire figure (suptitle).
    subtitles : list of list of str
        Nested list of subplot titles, same shape as `ds`. `subtitles[i][j]` is the title
        for subplot (i, j).
    nbins : int
        Number of bins to use in each histogram.
    xlabel : str
        Label for the x-axis of each subplot.
    unit : str
        Unit string to display in statistical annotations (e.g., 'K', 'mm').
    fname : str
        Path (including filename) where the figure will be saved.
    size : tuple of float
        Figure size in inches, e.g., (width, height).
    xlim : tuple of float, optional
        x-axis limits as (xmin, xmax). Default is None (auto-scaled).
    ylim : tuple of float, optional
        y-axis limits as (ymin, ymax). Default is None (auto-scaled).

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.
	"""
	# Determine number of rows.
	nrows = len(ds)
	ncols = len(ds[0])

	# Create plot space.
	fig, axs = plt.subplots(nrows, ncols, dpi=300, tight_layout=True)

	# Loop through datasets.
	for i in range(nrows):
		for j in range(ncols):
			# Create histogram.
			axs[i][j].hist(ds[i][j].data.flatten(), bins=nbins)

			# Add subplot title.
			axs[i][j].set_title(subtitles[i][j])

			# Set axis label.
			axs[i][j].set_xlabel(xlabel)

			# Set axes limit.
			if xlim != None:
				# x-axis.
				axs[i][j].set_xlim(xlim)
			if ylim != None:
				# y-axis.
				axs[i][j].set_ylim(xlim)

			# Remove ytick labels.
			axs[i][j].set_yticklabels([])

			# Add statistical information.
			add_statistical_information(ds[i][j].data, unit, axs[i][j])

	# Add title.
	fig.suptitle(title, weight='bold')

	# Change size.
	fig.set_size_inches(size)

	# Save result.
	plt.savefig(fname)

	# Output information.
	print("Created distribution successfully")

	return True

def plot_series(cubes, operation, title, labels, ylabel, fname):
	"""
    Plot time series of spatially aggregated Iris cubes.

    Each input cube is collapsed over latitude and longitude using the
    specified aggregation operation, producing a single time series per cube.
    The resulting series are plotted together with labels, a title, and y-axis
    label, and the figure is saved to file.

    Parameters
    ----------
    cubes : list of iris.cube.Cube
        List of Iris cubes, each containing a ``time`` coordinate and 
        2D spatial coordinates ``latitude`` and ``longitude``.
    operation : function
        Aggregation function used in the collapse, e.g. 
        ``iris.analysis.MEAN`` or ``iris.analysis.SUM``.
    title : str
        Title for the plot.
    labels : list of str
        Labels for the plotted time series, one per cube.
    ylabel : str
        Label for the y-axis (e.g., 'Temperature (K)', 'Precipitation (mm/day)').
    fname : str
        Path (including filename) where the figure will be saved.

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.

    Notes
    -----
    - The series are collapsed spatially over ``['longitude', 'latitude']``.
    - The x-axis is formatted for dates automatically.
    - The output is saved at 300 dpi.
    """
	# Determine number of rows.
	nrows = len(cubes)

	# Loop through datasets.
	for i in range(nrows):
		# Calculate mean spatial mean at each timestep.
		series_cube = cubes[i].collapsed(
			['longitude', 'latitude'], operation
		)

		# Create plot.
		iplt.plot(series_cube, label=labels[i])

	# Deal with x-axis ticks.
	plt.gcf().autofmt_xdate()

	# Set axis label.
	plt.ylabel(ylabel)

	# Put a grid on the plot.
	plt.grid(True)

	# Add legend.
	plt.legend(loc=0)

	# Add title.
	plt.title(title, weight='bold')

	# Save result.
	plt.savefig(fname, dpi=300)

	return True

def plot_diurnal_cycle(cubes, operation, title, labels, ylabel, fname):
	"""
    Plot the diurnal cycle of one or more Iris cubes.

    For each cube, the function bins data by hour of the day (from the ``time``
    coordinate), applies the specified aggregation operation within each hour,
    collapses over spatial dimensions, and plots the resulting hourly cycle.
    The figure is saved to file.

    Parameters
    ----------
    cubes : list of iris.cube.Cube
        List of Iris cubes, each containing a ``time`` coordinate and
        2D spatial coordinates ``latitude`` and ``longitude``.
    operation : function
        Aggregation function used for both hourly grouping and spatial 
        collapse, e.g. ``iris.analysis.MEAN``, ``iris.analysis.SUM``,
        ``iris.analysis.MAX``, etc.
    title : str
        Title for the plot.
    labels : list of str
        Labels for each plotted diurnal cycle, one per cube.
    ylabel : str
        Label for the y-axis (e.g., 'Temperature (K)', 'Precipitation (mm/hr)').
    fname : str
        Path (including filename) where the figure will be saved.

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.

    Notes
    -----
    - The ``hour`` coordinate is added dynamically via 
      :func:`iris.coord_categorisation.add_hour`.
    - Each cube is collapsed spatially over ``['latitude', 'longitude']`` 
      using the provided `operation`.
    - The x-axis runs from 0–23 hours.
    - The figure is saved at 300 dpi.
    """
	# Determine number of rows.
	nrows = len(cubes)

	# Get timesteps for x-axis.
	cube_time = cubes[0].coord("time")
	steps = cube_time.units.num2date(cube_time.points)

	# Loop through datasets.
	for i in range(nrows):
		# Get cube.
		cube = cubes[i]

		# Add an 'hour' coordinate to the cube
		iris.coord_categorisation.add_hour(cube, "time", name="hour")

		# Calculate mean spatial mean at each timestep.
		diurnal_cycle = cube.aggregated_by("hour", operation)

		# Collapse spatial dimensions to get mean temperature for each hour
		diurnal_operation = diurnal_cycle.collapsed(
			['latitude', 'longitude'], operation
		)
		
		# Get time.
		t = diurnal_operation.coord("hour").points

		# Create plot.
		plt.plot(t, diurnal_operation.data, label=labels[i])

	# Set axis labels.
	plt.xlabel("Hour")
	plt.ylabel(ylabel)

	# Add legend.
	plt.legend(loc=0)

	# Add title.
	plt.title(title, weight='bold')

	# Save result.
	plt.savefig(fname, dpi=300)

	return True

def find_nearest_grid_point(cube, target_lat, target_lon, max_search_radius=5):
	"""
    Extract the time series at the nearest *valid* grid point to a target lat/lon.

    The function first finds the nearest latitude/longitude indices to the target.
    If the corresponding grid cell has valid data (not masked/NaN) in the first
    time slice, it returns that time series. Otherwise, it searches within a
    square neighborhood around the initial index (up to `max_search_radius`
    grid cells away in each direction) and returns the time series at the closest
    valid grid cell to the target coordinates.

    Parameters
    ----------
    cube : iris.cube.Cube
        Data cube with dimensions ordered as ``(time, latitude, longitude)``.
        The first time slice is used to determine validity (mask/NaN).
    target_lat : float
        Target latitude in degrees north.
    target_lon : float
        Target longitude in degrees east (same convention/range as the cube).
    max_search_radius : int, optional
        Maximum search radius **in grid cells** around the initial nearest
        index, used only if the initial point is invalid. Default is ``5``.

    Returns
    -------
    time_series : iris.cube.Cube
        A 1D cube (over ``time``) extracted at the selected grid cell.

    Raises
    ------
    ValueError
        If no valid grid point is found within the search neighborhood.

    Notes
    -----
    - Validity is checked on the **first** time slice only.
    - Distance is computed in index space using a simple Euclidean metric on
      lat/lon degrees (``np.hypot``). For small neighborhoods this is adequate.
    - `max_search_radius` is in **index units**, not degrees. For example,
      if your grid spacing is 0.1°, a radius of 5 covers ±0.5° around the
      initial nearest-neighbor index.
    """
	# Get longitude and latitude array.
	lats = cube.coord('latitude').points
	lons = cube.coord('longitude').points

	# Get initial index of nearest neighbour.
	lat_idx = np.abs(lats - target_lat).argmin()
	lon_idx = np.abs(lons - target_lon).argmin()

	# Get the first time slice as data.
	data = cube[0].data

    # Check if that point has valid data
	if not np.ma.is_masked(data[lat_idx, lon_idx]) and not np.isnan(data[lat_idx, lon_idx]):
		return cube[:, lat_idx, lon_idx]

    # Otherwise, search within a neighbourhood
	lat_range = range(
		max(0, lat_idx - max_search_radius), min(len(lats), lat_idx + max_search_radius + 1)
	)
	lon_range = range(
		max(0, lon_idx - max_search_radius), min(len(lons), lon_idx + max_search_radius + 1)
	)
	
	# Set minimum distance starting point.
	min_dist = float('inf')
	best_slice = None

	# Loop through longitude and latitude range.
	for i in lat_range:
		for j in lon_range:
			# Check if data is masked or NaN at this new point.
			if not np.ma.is_masked(data[i, j]) and not np.isnan(data[i, j]):
				# If satisifed, calculated distance between target and valid point.
				dist = np.hypot(lats[i] - target_lat, lons[j] - target_lon)

				# If the grid point is closer, save this point.
				if dist < min_dist:
					min_dist = dist
					best_slice = cube[:, i, j]

	# Only return if there is a valid grid point.
	if best_slice is not None:
		return best_slice
	else:
		raise ValueError("No valid grid point found near the target location.")

def plot_location_series(cubes, title, labels, location_name, unit, fname, obs=None):
	"""
    Plot time series from multiple Iris cubes at the grid point nearest to a named location.

    The function uses geocoding (via Nominatim) to resolve a place name into
    latitude/longitude, finds the nearest valid grid point in each cube,
    extracts the corresponding time series, and plots them together. An
    optional set of observational data can also be plotted for comparison.

    Parameters
    ----------
    cubes : list of iris.cube.Cube
        List of Iris cubes, each with dimensions including time, latitude, and longitude.
    title : str
        Title for the plot.
    labels : list of str
        Labels for each cube's plotted time series, one per cube.
    location_name : str
        Place name to resolve to coordinates via geopy's Nominatim service.
        (Requires internet access.)
    unit : str
        Label for the y-axis (e.g., 'Temperature (K)', 'Precipitation (mm/day)').
    fname : str
        Path (including filename) where the figure will be saved.
    obs : tuple of (array_like, array_like), optional
        Observational data to plot as ``(times, values)``.
        - ``obs[0]`` : sequence of datetime-like values for the x-axis.
        - ``obs[1]`` : sequence of numerical values for the y-axis.
        Default is None (no observations plotted).

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.

    Raises
    ------
    Exception
        If the `location_name` cannot be resolved to coordinates.
    ValueError
        If no valid grid point is found near the resolved location in a cube.

    Notes
    -----
    - Location lookup uses OpenStreetMap's Nominatim service. Ensure network
      access is available when running this function.
    - Grid point selection relies on :func:`find_nearest_grid_point`.
    - All time series are plotted on the same axis with a common y-label (`unit`).
    - Observations, if provided, are added as an additional line labelled
      "Observations".
    """
	# Determine number of rows.
	nrows = len(cubes)

	# Create a geolocator object with a user agent (can be any name)
	geolocator = Nominatim(user_agent="cocathasaigh")

	# Enter place name
	location = geolocator.geocode(location_name)

	# Check if location was found.
	if not location:
		raise Exception("Location not found.")

	# Loop through datasets.
	for i in range(nrows):
		# Retrieve cube point closet to location.
		nearest_cube_point = find_nearest_grid_point(
			cube=cubes[i], 
			target_lat=location.latitude, 
			target_lon=location.longitude
		)

		# Plot point.
		iplt.plot(nearest_cube_point, label=labels[i])

	# Deal with x-axis ticks.
	plt.gcf().autofmt_xdate()
	
	# Set axis label.
	plt.ylabel(unit)

	# Add observations.
	if obs is not None:
		plt.plot(obs[0], obs[1], label="Observations")

	# Add title.
	plt.title("{}\n{}".format(title, location_name), weight='bold')

	# Add legend.
	plt.legend(loc=0)

	# Add grid.
	plt.grid(True)

	# Save result.
	plt.savefig(fname, dpi=300)

	return True

def rainfall_cmap():
	"""
    Construct a custom colormap and contour levels for rainfall visualisation.

    The function defines a sequence of precipitation thresholds (in mm)
    and a corresponding set of RGB colors. A `ListedColormap` is built
    and returned together with the contour levels.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap for rainfall, labelled `"precipitation"`.
    clevs : list of float
        Contour levels (mm) to be used with the colormap.

    Raises
    ------
    AssertionError
        If the number of contour levels and colors are inconsistent.

    Notes
    -----
    - Thresholds (`clevs`) include:
      [0, 0.1, 1, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 250].
    - Colors progress from white (low values) through cyan, green, yellow,
      orange, red, magenta, purple, gray, to beige (high values).
    - Designed for use in filled contour plots of accumulated precipitation.
    """
	# Draw filled contours.
	clevs = [
		0, 0.1, 1, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 250
	]
	
	# Color map data.
	cmap_data = [
		(1.0, 1.0, 1.0),
		(0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
		(0.0, 1.0, 1.0),
		(0.0, 0.8784313797950745, 0.501960813999176),
		(0.0, 0.7529411911964417, 0.0),
		(0.501960813999176, 0.8784313797950745, 0.0),
		(1.0, 1.0, 0.0),
		(1.0, 0.6274510025978088, 0.0),
		(1.0, 0.0, 0.0),
		(1.0, 0.125490203499794, 0.501960813999176),
		(0.9411764740943909, 0.250980406999588, 1.0),
		(0.501960813999176, 0.125490203499794, 1.0),
		(0.250980406999588, 0.250980406999588, 1.0),
		(0.125490203499794, 0.125490203499794, 0.501960813999176),
		(0.125490203499794, 0.125490203499794, 0.125490203499794),
		(0.501960813999176, 0.501960813999176, 0.501960813999176),
		(0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
		(0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
		#(0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
		#(0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
		#(0.4000000059604645, 0.20000000298023224, 0.0)
	]

	# Ensure the number of levels and the number of colours are equal.
	assert len(clevs) == len(cmap_data), "Mismatch between the numbers of colours and numbers of levels"

	# Create colour map.
	cmap = colors.ListedColormap(cmap_data, "precipitation")

	return cmap, clevs

def prate_1hr_cmap():
	"""
    Construct a custom colormap and contour levels for 1-hourly precipitation rate.

    The function defines a sequence of precipitation rate thresholds (in mm/hr)
    and a corresponding set of RGB colors. A `ListedColormap` is built and
    returned together with the contour levels.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap for precipitation rate, labelled `"precipitation_rate"`.
    clevs : list of float
        Contour levels (mm/hr) to be used with the colormap.

    Raises
    ------
    AssertionError
        If the number of contour levels and colors are inconsistent.

    Notes
    -----
    - Thresholds (`clevs`) include:
      [0, 0.2, 0.5, 1, 1.5, 2, 4, 10, 25, 50, 100].
    - Colors progress from white (no rain) through shades of cyan/blue,
      then magenta/orange, to red (intense rainfall).
    - Designed for use in filled contour plots of precipitation rate fields.
    """
	# Draw filled contours.
	clevs = [
		0, 0.2, 0.5, 1, 1.5, 2, 4, 10, 25, 50, 100
	]
	
	# Color map data.
	cmap_data = [
		(1.0, 1.0, 1.0),
		(0.0, 1.0, 1.0),
		(0.0, 0.85, 1.0),
		(0.0, 0.73, 1.0),
		(0.0, 0.61, 1.0),
		(0.0, 0.49, 1.0),
		(0.0, 0.0, 1.0),
		(0.7, 0.0, 1.0),
		(1.0, 0.0, 1.0),
		(1.0, 0.5, 0.0),
		(1.0, 0.0, 0.0),
	]

	# Ensure the number of levels and the number of colours are equal.
	assert len(clevs) == len(cmap_data), "Mismatch between the numbers of colours and numbers of levels"

	# Create colour map.
	cmap = colors.ListedColormap(cmap_data, "precipitation_rate")

	return cmap, clevs

def create_param_plot(cubes, title, subtitles, levels, unit, fname, size):
	"""
    Create a grid of filled contour plots for meteorological parameters.

    Each subplot corresponds to one Iris cube, displayed on a
    Plate Carrée map projection with coastlines and a title.
    The colormap and contour levels are chosen automatically
    based on the physical unit (e.g., rainfall, temperature, pressure).

    Parameters
    ----------
    cubes : list of list of iris.cube.Cube
        Nested list (2D structure) of Iris cubes to plot. Each cube must
        contain latitude and longitude coordinates.
    title : str
        Overall title for the figure (added as a suptitle).
    subtitles : list of list of str
        Nested list of subplot titles, same shape as `cubes`.
    levels : list of float
        Contour levels to use for plotting (may be overridden depending
        on the chosen `unit`).
    unit : str
        Physical unit of the parameter (e.g., 'mm', 'mm/hr', 'K', 'hPa', 'm s-1').
        Determines the colormap and normalisation.
    fname : str
        Path (including filename) where the figure will be saved.
    size : tuple of float
        Figure size in inches, e.g., (width, height).

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.

    Notes
    -----
    - For rainfall ('mm') and precipitation rate ('mm/hr'), custom
      colormaps are retrieved from `rainfall_cmap()` and
      `prate_1hr_cmap()`, respectively, and a `BoundaryNorm` is applied.
    - For other parameters:
      
      * 'K' → ``RdYlBu_r`` (temperature)  
      * 'hPa' → ``rainbow`` (pressure; contour levels downsampled by 3)  
      * 'm s-1' → ``jet`` (wind speed)  
      * everything else → ``viridis``  

    - A horizontal colorbar is added below the subplots with the unit
      as the label.
    - Coastlines are drawn with 10m resolution.
    - If the longitude/latitude coordinates are already 2D (curvilinear grid),
      they are used directly instead of creating a meshgrid.
	"""
	# Determine number of rows.
	nrows = len(cubes)
	ncols = len(cubes[0])

	# Create plot space.
	fig, axs = plt.subplots(
		nrows, ncols, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	if unit == "mm":
		# Retrieve rainfall levels and create normalisation.
		# Retrieve colour map.
		cmap, levels = rainfall_cmap()
		# Create norm.
		norm = colors.BoundaryNorm(levels, cmap.N)
		# Add to list.
		cmapping = [levels, cmap, norm]
	elif unit == "mm/hr":
		# Retrieve rainfall levels and create normalisation.
		# Retrieve colour map.
		cmap, levels = prate_1hr_cmap()
		# Create norm.
		norm = colors.BoundaryNorm(levels, cmap.N)
		# Add to list.
		cmapping = [levels, cmap, norm]
	else:
		# Define colour map for other parameters.
		if unit == "K":
			# Air temperature.
			cmap = "RdYlBu_r"
		elif unit == "hPa":
			# Air pressure.
			cmap = "rainbow"

			# Cut the number of contourf levels.
			levels = levels[::3]
		elif unit == "m s-1":
			cmap = "jet"
		else:
			# Everything else.
			cmap = "viridis"

		# Add to list.
		cmapping = [levels, cmap]

	# Loop through datasets.
	for i in range(nrows):
		cubes_row = []
		for j in range(ncols):
			# Create longitude-latitude meshgrid.
			lon = cubes[i][j].coord('longitude').points
			lat = cubes[i][j].coord('latitude').points
			x, y = np.meshgrid(lon, lat)

			# Check if the grid is rectangular. 
			if lon.ndim == 1:
				# Generate mesh grid.
				x, y = np.meshgrid(lon, lat)
			else:
				# Get longitude-latitude as x-y.
				x, y = lon, lat

			# Get axis for plotting.
			if nrows == 1 and ncols == 1:
				ax = axs
			elif nrows == 1:
				ax = axs[j]
			elif ncols == 1:
				ax = axs[i]
			else:
				ax = axs[i][j]

			# Add high-resolution coastlines.
			ax.coastlines(resolution='10m')

			# Add subplot title.
			ax.set_title(subtitles[i][j])

			# Create filled contour plot.
			contourf = ax.contourf(
				x, y, cubes[i][j].data, levels=cmapping[0], cmap=cmapping[1], extend="both" 
			)

	# Add title.
	fig.suptitle("{}".format(title), weight='bold')

	# Create colour bar.
	cbar = plt.colorbar(
		contourf, 
		ax=fig.axes, 
		aspect=60, 
		orientation="horizontal", 
		extend="both", 
		fraction=0.02
	)
	cbar.set_label(unit)
	cbar.ax.tick_params(length=0)

	# Set size.
	fig.set_size_inches(size)

	# Save file.
	plt.savefig(fname)

	return True

def create_bias_plot(cubes, title, subtitles, levels, unit, fname, size):
	"""
    Create a grid of spatial bias plots with statistical annotations.

    Each subplot corresponds to one Iris cube and is displayed on a Plate
    Carrée map projection with coastlines and a title. A diverging colormap
    is used to highlight positive/negative biases, and summary statistics
    are added to each subplot.

    Parameters
    ----------
    cubes : list of list of iris.cube.Cube
        Nested list (2D structure) of Iris cubes to plot. Each cube must
        contain latitude and longitude coordinates.
    title : str
        Overall title for the figure (added as a suptitle).
    subtitles : list of list of str
        Nested list of subplot titles, same shape as `cubes`.
    levels : list of float
        Contour levels to use for filled contour plotting.
    unit : str
        Physical unit of the parameter (e.g., 'mm', 'K', 'hPa').
        Determines the choice of colormap:
        
        - "mm" → ``BrBG`` (precipitation biases)  
        - everything else → ``RdBu_r`` (diverging red/blue)  
    fname : str
        Path (including filename) where the figure will be saved.
    size : tuple of float
        Figure size in inches, e.g., (width, height).

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.

    Notes
    -----
    - Bias values are expected in the input cubes (e.g., model minus observations).
    - Each subplot includes anchored statistics (mean absolute error,
      standard deviation, and mean bias) via
      :func:`add_statistical_information`.
    - A horizontal colorbar is added below the subplots, labelled with `unit`.
    - Coastlines are drawn with 10m resolution.
    - The function uses :func:`iris.plot.contourf` (`iplt.contourf`) for plotting,
      which directly accepts an Iris cube.
    """
	# Determine number of rows.
	nrows = len(cubes)
	ncols = len(cubes[0])

	# Create plot space.
	fig, axs = plt.subplots(
		nrows, ncols, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	# Set colour map.
	if unit == "mm":
		# Precipitation.
		cmap = "BrBG"
	else:
		# Everything else:
		cmap= "RdBu_r"

	# Loop through datasets.
	for i in range(nrows):
		cubes_row = []
		for j in range(ncols):
			# # Create longitude-latitude meshgrid.
			# lon = cubes[i][j].coord('longitude').points
			# lat = cubes[i][j].coord('latitude').points
			# x, y = np.meshgrid(lon, lat)

			# # Check if the grid is rectangular. 
			# if lon.ndim == 1:
			# 	# Generate mesh grid.
			# 	x, y = np.meshgrid(lon, lat)
			# else:
			# 	# Get longitude-latitude as x-y.
			# 	x, y = lon, lat

			# Get axis for plotting.
			if nrows == 1 and ncols == 1:
				ax = axs
			elif nrows == 1:
				ax = axs[j]
			elif ncols == 1:
				ax = axs[i]
			else:
				ax = axs[i][j]

			# Add high-resolution coastlines.
			ax.coastlines(resolution='10m')

			# Add subplot title.
			ax.set_title(subtitles[i][j])

			# Create filled contour plot.
			contourf = iplt.contourf(
				cubes[i][j], ax, levels=levels, cmap=cmap, extend="both" 
			)

			# Add statistical information.
			add_statistical_information(cubes[i][j].data, unit, ax)

	# Add title.
	fig.suptitle("{}".format(title), weight='bold')

	# Create colour bar.
	cbar = plt.colorbar(
		contourf, 
		ax=fig.axes, 
		aspect=60, 
		orientation="horizontal", 
		extend="both", 
		fraction=0.02
	)
	cbar.set_label(unit)
	cbar.ax.tick_params(length=0)

	# Set size.
	fig.set_size_inches(size)

	# Save file.
	plt.savefig(fname)

	return True

def create_param_ani(cubes, focus_period, title, subtitles, unit, folder, size, interval):
	"""
    Create an animated sequence of meteorological parameter maps.

    Each subplot corresponds to one Iris cube, displayed on a Plate
    Carrée map projection with coastlines. The function extracts data for
    a given focus period, creates filled contour maps for each timestep,
    and generates both individual frame images (PNG) and an MP4 animation.

    Parameters
    ----------
    cubes : list of list of iris.cube.Cube
        Nested list (2D structure) of Iris cubes to animate. Each cube
        must contain dimensions including ``time``, ``latitude``, and
        ``longitude``.
    focus_period : iris.Constraint
        Constraint selecting the time range to animate.
    title : str
        Overall title for the animation. The current date is appended
        automatically in each frame.
    subtitles : list of list of str
        Nested list of subplot titles, same shape as `cubes`.
    unit : str
        Unit of the parameter being plotted. Determines color mapping:
        
        - "mm" → rainfall colormap via :func:`rainfall_cmap`
        - "mm/hr" → precipitation rate colormap via :func:`prate_1hr_cmap`
        - "K" → ``RdYlBu_r``
        - "hPa" → ``rainbow`` (pressure, with optional contours)
        - everything else → ``viridis``
    folder : str
        Path to the output folder. Individual frames (PNG) and the MP4
        animation will be saved here. The folder is created if it does
        not exist.
    size : tuple of float
        Figure size in inches, e.g., (width, height).
    interval : int
        Delay between frames in milliseconds (passed to
        :class:`matplotlib.animation.FuncAnimation`).

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the animation.

    Notes
    -----
    - Each frame includes a title showing the simulation title and the
      current datetime.
    - If ``"Reference: ERA5"`` is in the title, only every fourth frame
      is saved as PNG; otherwise, all frames are saved.
    - Colorbars are added once at the bottom of the figure, labelled with `unit`.
    - For pressure fields (`unit="hPa"`), contour lines with labels are
      added on top of the filled contours.
    - Progress is tracked with a :class:`tqdm` progress bar.
    - The final MP4 is saved as ``<folder>/<foldername>_ani.mp4`` where
      ``foldername`` is the basename of `folder`.
    """
	# Determine number of rows.
	nrows = len(cubes)
	ncols = len(cubes[0])

	# Create plot space.
	fig, axs = plt.subplots(
		nrows, ncols, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	if unit == "mm":
		# Retrieve rainfall levels and create normalisation.
		# Retrieve colour map.
		cmap, levels = rainfall_cmap()
		# Create norm.
		norm = colors.BoundaryNorm(levels, cmap.N)
		# Add to list.
		cmapping = [levels, cmap, norm]
	elif unit == "mm/hr":
		# Retrieve rainfall levels and create normalisation.
		# Retrieve colour map.
		cmap, levels = prate_1hr_cmap()
		# Create norm.
		norm = colors.BoundaryNorm(levels, cmap.N)
		# Add to list.
		cmapping = [levels, cmap, norm]
	else:
		# Define levels based on reference dataset.
		# Get maximum.
		min_param = np.floor(
			cubes[0][0].collapsed(
				["time", "longitude", "latitude"], iris.analysis.MIN
			).data
		)
		# Get minimum.
		max_param = np.ceil(
			cubes[0][0].collapsed(
				["time", "longitude", "latitude"], iris.analysis.MAX
			).data
		)
		# Convert to integers.
		min_param, max_param = int(min_param), int(max_param)
		# Create levels.
		levels = np.rint(
			np.linspace(min_param, max_param, (max_param - min_param) + 1)
		)

		# Define colour map for other parameters.
		if unit == "K":
			# Air temperature.
			cmap = "RdYlBu_r"
		elif unit == "hPa":
			# Air pressure.
			cmap = "rainbow"

			# Cut the number of contourf levels.
			levels = levels[::3]
		else:
			# Everything else.
			cmap = "viridis"

		# Add to list.
		cmapping = [levels, cmap]

	# Loop through datasets.
	contourfs = []
	new_cubes = []
	for i in range(nrows):
		cubes_row = []
		for j in range(ncols):
			# Create longitude-latitude meshgrid.
			lon = cubes[i][j].coord('longitude').points
			lat = cubes[i][j].coord('latitude').points
			x, y = np.meshgrid(lon, lat)

			# Check if the grid is rectangular. 
			if lon.ndim == 1:
				# Generate mesh grid.
				x, y = np.meshgrid(lon, lat)
			else:
				# Get longitude-latitude as x-y.
				x, y = lon, lat

			# Get axis for plotting.
			if nrows == 1 and ncols == 1:
				ax = axs
			elif nrows == 1:
				ax = axs[j]
			elif ncols == 1:
				ax = axs[i]
			else:
				ax = axs[i][j]

			# Add high-resolution coastlines.
			ax.coastlines(resolution='10m')

			# Add subplot title.
			ax.set_title(subtitles[i][j])

			# Extract time period.
			cube = cubes[i][j].extract(focus_period)
			cubes_row.append(cube)

			# Create inital contour plot.
			if len(cmapping) == 3:
				# Rainfall.
				contourf = ax.contourf(
					x, y, cube[0].data, 
					levels=cmapping[0], cmap=cmapping[1], norm=cmapping[2], 
					extend="max"
				)
			else:
				# Other parameters.
				contourf = ax.contourf(
					x, y, cube[0].data, 
					levels=cmapping[0], cmap=cmapping[1] 
				)
			contourfs.append(contourf)

		# Add row of time constrained cubes.
		new_cubes.append(cubes_row)

	# Redefine cubes variable.
	cubes = new_cubes
	
	# Update function for animation.
	def update(frame, cubes, title, levels, pbar, folder, cmapping):
		# Determine number of rows.
		nrows = len(cubes)
		ncols = len(cubes[0])

		# Remove plots.
		# Define global.
		global contours
		global contourfs
		# Try removal.
		try:
			for contourf in contourfs:
				contourf.remove()
			for contour in contours:
				contour.remove()
		except:
			pass

		# Determine date.
		cube_time = cubes[0][0].coord("time")
		date = cube_time.units.num2date(cube_time.points)[frame]

		# Loop through cubes.
		contours = []
		contourfs = []
		for i in range(nrows):
			for j in range(ncols):
				# Create longitude-latitude meshgrid.
				lon = cubes[i][j].coord('longitude').points
				lat = cubes[i][j].coord('latitude').points

				# Check if the grid is rectangular. 
				if lon.ndim == 1:
					# Generate mesh grid.
					x, y = np.meshgrid(lon, lat)
				else:
					# Get longitude-latitude as x-y.
					x, y = lon, lat

				# Get relevant cube.
				cube_data = cubes[i][j][frame].data

				# Get axis for plotting.
				if nrows == 1 and ncols == 1:
					ax = axs
				elif nrows == 1:
					ax = axs[j]
				elif ncols == 1:
					ax = axs[i]
				else:
					ax = axs[i][j]

				# Create inital contour plot.
				if len(cmapping) == 3:
					# Rainfall.
					contourf = ax.contourf(
						x, y, cube_data, 
						levels=cmapping[0], cmap=cmapping[1], norm=cmapping[2], 
						extend="max"
					)
				else:
					# Every other parameter.
					contourf = ax.contourf(
						x, y, cube_data, 
						levels=cmapping[0], cmap=cmapping[1], extend="both"
					)

					if unit == "hPa":
						# Plot contour lines if looking at air pressure.
						contour = ax.contour(
							x, y, cube_data, 
							levels=cmapping[0], linewidths=0.5, colors='k'
						)

						# Add labels.
						ax.clabel(contour, inline=True, fontsize=8)

						# Append to list.
						contours.append(contour)
				contourfs.append(contourf)

		# Add title.
		fig.suptitle(
			"{}\nDate: {}".format(title, date.strftime('%d-%m-%Y %H:%M')), 
			weight='bold'
		)

		# Save frame.
		fname = "{}_{}.png".format(
			folder.rsplit('/', 1)[-1], date.strftime('%Y%m%d%H%M')
		)

		if not "Reference: ERA5" in title:
			# Save every frame is gridded dataset.
			plt.savefig("{}/{}".format(folder, fname))
		else:
			# Save every fourth frame if ERA5.
			if frame % 4 == 0:
				plt.savefig("{}/{}".format(folder, fname))

		# Update progress bar.
		pbar.update(1)

		return contourfs

	# Initialize tqdm progress bar
	pbar = tqdm(
		total=cubes[i][j].shape[0], desc="Animating parameter plots"
	)

	# Create animation.
	ani = animation.FuncAnimation(
		fig, 
		update, 
		frames=cubes[i][j].shape[0], 
		fargs=(cubes, title, levels, pbar, folder, cmapping),
		interval=interval
	)

	# Create colour bar.
	cbar = plt.colorbar(
		contourf, 
		ax=fig.axes, 
		aspect=60, 
		orientation="horizontal", 
		extend="both", 
		fraction=0.02
	)
	cbar.set_label(unit)
	cbar.ax.tick_params(length=0)

	# Set size.
	fig.set_size_inches(size)

	# Check if folder exists.
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Save file as MP4.
	fname = "{}_ani.mp4".format(folder.rsplit('/', 1)[-1])
	ani.save(filename="{}/{}".format(folder, fname))

	# Close progress bar.
	pbar.close()

	return True

def create_bias_ani(cubes, focus_period, title, subtitles, levels, unit, folder, size, interval, stat_info=True):
	"""
    Create an animated sequence of bias plots.

    Each subplot corresponds to one Iris cube, displayed on a Plate
    Carrée map projection with coastlines. The function extracts data
    for a given focus period, creates filled contour maps for each timestep,
    and generates both individual frame images (PNG) and an MP4 animation.
    Optional summary statistics can be added to each subplot.

    Parameters
    ----------
    cubes : list of list of iris.cube.Cube
        Nested list (2D structure) of Iris cubes to animate. Each cube must
        contain dimensions including ``time``, ``latitude``, and ``longitude``.
        The data are expected to represent model–observation differences
        (biases).
    focus_period : iris.Constraint
        Constraint selecting the time range to animate.
    title : str
        Overall title for the animation. The current date is appended
        automatically in each frame.
    subtitles : list of list of str
        Nested list of subplot titles, same shape as `cubes`.
    levels : list of float
        Contour levels for bias plotting.
    unit : str
        Unit of the variable (e.g., "mm", "mm/hr", "K"). Determines
        the colormap:

        - "mm" or "mm/hr" → ``BrBG`` (precipitation biases)
        - everything else → ``RdBu_r`` (diverging red/blue)
    folder : str
        Path to the output folder. Individual frames (PNG) and the MP4
        animation will be saved here. The folder is created if it does
        not exist.
    size : tuple of float
        Figure size in inches, e.g., (width, height).
    interval : int
        Delay between frames in milliseconds (passed to
        :class:`matplotlib.animation.FuncAnimation`).
    stat_info : bool, optional
        If True (default), statistical summaries (mean absolute value,
        standard deviation, mean bias) are added to each subplot using
        :func:`add_statistical_information`.

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the animation.

    Notes
    -----
    - Each frame includes a title showing the simulation title and the
      current datetime.
    - If ``"Reference: ERA5"`` is in the title, only every fourth frame
      is saved as PNG; otherwise, all frames are saved.
    - A horizontal colorbar is added below the subplots, labelled with `unit`.
    - Coastlines are drawn with 10m resolution.
    - Progress is tracked with a :class:`tqdm` progress bar.
    - The final MP4 is saved as ``<folder>/<foldername>_ani.mp4`` where
      ``foldername`` is the basename of `folder`.
    """
	# Determine number of rows.
	nrows = len(cubes)
	ncols = len(cubes[0])

	# Create plot space.
	fig, axs = plt.subplots(
		nrows, ncols, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	# Set colour map.
	if unit == "mm" or unit == "mm/hr":
		# Precipitation.
		cmap = "BrBG"
	else:
		# Everything else:
		cmap= "RdBu_r"

	# Loop through datasets.
	contourfs = []
	new_cubes = []
	for i in range(nrows):
		cubes_row = []
		for j in range(ncols):
			# Create longitude-latitude meshgrid.
			lon = cubes[i][j].coord('longitude').points
			lat = cubes[i][j].coord('latitude').points
			x, y = np.meshgrid(lon, lat)

			# Check if the grid is rectangular. 
			if lon.ndim == 1:
				# Generate mesh grid.
				x, y = np.meshgrid(lon, lat)
			else:
				# Get longitude-latitude as x-y.
				x, y = lon, lat

			# Get axis for plotting.
			if nrows == 1 and ncols == 1:
				ax = axs
			elif nrows == 1:
				ax = axs[j]
			elif ncols == 1:
				ax = axs[i]
			else:
				ax = axs[i][j]

			# Add high-resolution coastlines.
			ax.coastlines(resolution='10m')

			# Add subplot title.
			ax.set_title(subtitles[i][j])

			# Extract time period.
			cube = cubes[i][j].extract(focus_period)
			cubes_row.append(cube)

			# Create inital contour plot.
			contourf = ax.contourf(
				x, y, cube[0].data, levels=levels, cmap=cmap, extend="both"
			)
			contourfs.append(contourf)

			# Add statistical information.
			if stat_info:
				add_statistical_information(cubes[i][j].data, unit, ax)

		# Add row of time constrained cubes.
		new_cubes.append(cubes_row)

	# Redefine cubes variable.
	cubes = new_cubes
	
	# Update function for animation.
	def update(frame, cubes, title, levels, pbar, folder):
		# Determine number of rows.
		nrows = len(cubes)
		ncols = len(cubes[0])

		# Remove plots.
		# Define global.
		global contourfs
		# Try removal.
		try:
			for contourf in contourfs:
				contourf.remove()
		except:
			pass

		# Determine date.
		cube_time = cubes[0][0].coord("time")
		date = cube_time.units.num2date(cube_time.points)[frame]

		# Set colour map.
		if unit == "mm":
			# Precipitation.
			cmap = "BrBG"
		else:
			# Everything else:
			cmap= "RdBu_r"

		# Loop through cubes.
		contourfs = []
		for i in range(nrows):
			for j in range(ncols):
				# Create longitude-latitude meshgrid.
				lon = cubes[i][j].coord('longitude').points
				lat = cubes[i][j].coord('latitude').points
				x, y = np.meshgrid(lon, lat)

				# Check if the grid is rectangular. 
				if lon.ndim == 1:
					# Generate mesh grid.
					x, y = np.meshgrid(lon, lat)
				else:
					# Get longitude-latitude as x-y.
					x, y = lon, lat

				# Get axis for plotting.
				if nrows == 1 and ncols == 1:
					ax = axs
				elif nrows == 1:
					ax = axs[j]
				elif ncols == 1:
					ax = axs[i]
				else:
					ax = axs[i][j]

				# Create inital contour plot.
				contourf = ax.contourf(
					x, y, cubes[i][j][frame].data, levels=levels, cmap=cmap, extend="both"
				)
				contourfs.append(contourf)

		# Add title.
		fig.suptitle(
			"{}\nDate: {}".format(title, date.strftime('%d-%m-%Y %H:%M')), 
			weight='bold'
		)

		# Save frame.
		fname = "{}_{}.png".format(
			folder.rsplit('/', 1)[-1], date.strftime('%Y%m%d%H%M')
		)
		
		if not "Reference: ERA5" in title:
			# Save every frame is gridded dataset.
			plt.savefig("{}/{}".format(folder, fname))
		else:
			# Save every fourth frame if ERA5.
			if frame % 4 == 0:
				plt.savefig("{}/{}".format(folder, fname))

		# Update progress bar.
		pbar.update(1)

		return contourfs

	# Set size.
	fig.set_size_inches(size)

	# Initialize tqdm progress bar
	pbar = tqdm(
		total=cubes[i][j].shape[0], desc="Animating bias plots"
	)

	# Create animation.
	ani = animation.FuncAnimation(
		fig, 
		update, 
		frames=cubes[i][j].shape[0], 
		fargs=(cubes, title, levels, pbar, folder),
		interval=interval
	)

	# Create colour bar.
	cbar = plt.colorbar(
		contourf, 
		ax=fig.axes, 
		aspect=60, 
		orientation="horizontal", 
		extend="both", 
		fraction=0.02
	)
	cbar.set_label(unit)
	cbar.ax.tick_params(length=0)

	# Check if folder exists.
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Save file as MP4.
	fname = "{}_ani.mp4".format(folder.rsplit('/', 1)[-1])
	ani.save(filename="{}/{}".format(folder, fname))

	# Close progress bar.
	pbar.close()

	return True

def get_true_track(storm_name, t0, t1):
	# Define basin.
	basin = tracks.TrackDataset(basin='north_atlantic', source='hurdat')

	# Get storm.
	storm = basin.get_storm((storm_name,t0.year)).to_xarray().sel(
		time=slice(t0.strftime("%Y-%m-%d %H:%M"), t1.strftime("%Y-%m-%d %H:%M"))
	)

	return storm.lon.values, storm.lat.values

def calculate_crude_track(psl_cube, time_constraint=None):
	"""
    Estimate a crude extratropical cyclone track from sea-level pressure data.

    For each timestep in the input cube, the location of the minimum sea-level
    pressure is identified. The resulting sequence of longitude/latitude
    points provides an approximate storm track.

    Parameters
    ----------
    psl_cube : iris.cube.Cube
        Sea-level pressure cube with dimensions ``(time, latitude, longitude)``.
        Must include ``time``, ``latitude``, and ``longitude`` coordinates.
    time_constraint : iris.Constraint, optional
        Constraint to restrict the analysis to a subset of the time dimension.
        Default is None (use all available timesteps).

    Returns
    -------
    lon : numpy.ndarray
        1D array of longitudes (degrees east) at the minimum pressure location
        for each timestep.
    lat : numpy.ndarray
        1D array of latitudes (degrees north) at the minimum pressure location
        for each timestep.

    Notes
    -----
    - The track is “crude” because only the single grid point of minimum sea-level
      pressure is retained per timestep; no temporal smoothing or tracking
      heuristics are applied.
    - Uses :func:`iris.analysis.MIN` to collapse over latitude and longitude.
    - If multiple points share the same minimum value at a timestep, all are returned,
      which may result in more points than timesteps.
    """
	# Extract relevant period.
	if time_constraint != None:
		psl_cube = psl_cube.extract(time_constraint)
	
	# Calculate minimum pressure values.
	min_psl = psl_cube.collapsed(
		["longitude", "latitude"], iris.analysis.MIN
	).data + np.zeros(
		(psl_cube.shape[1], psl_cube.shape[2], psl_cube.shape[0])
	)
	min_psl = np.transpose(min_psl, (2, 0, 1))
	# Get indices.
	min_psl_indx = np.where(psl_cube.data == min_psl)

	# Get relevant points.
	lon = psl_cube[min_psl_indx].coord("longitude").points
	lat = psl_cube[min_psl_indx].coord("latitude").points

	return lon, lat

def plot_extratropical_cyclone_track(cubes, focus_t0, focus_t1, title, labels, fname):
	"""
    Plot extratropical cyclone tracks from one or more simulations.

    Each input cube is constrained to a focus period, a crude track
    (longitude/latitude sequence) is calculated, and the resulting tracks
    are plotted on a Plate Carrée map with coastlines and land shading.
    Multiple simulations can be shown together with distinct colors.

    Parameters
    ----------
    cubes : list of iris.cube.Cube
        List of pressure (e.g., sea-level pressure) cubes from different
        simulations. Each cube must contain time, latitude, and longitude
        coordinates.
    focus_t0 : datetime-like
        Start of the focus period (inclusive).
    focus_t1 : datetime-like
        End of the focus period (inclusive).
    title : str
        Title for the plot.
    labels : list of str
        Labels for each cyclone track, one per input cube.
    fname : str
        Path (including filename) where the figure will be saved.

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the plot.

    Notes
    -----
    - Tracks are derived using :func:`calculate_crude_track`, which should
      return longitude and latitude arrays describing the storm path.
    - Only the period between `focus_t0` and `focus_t1` is plotted.
    - Land is shaded in beige (``#FFE6B3``), with coastlines drawn at 10m
      resolution.
    - Colors cycle through a standard list
      (``['b', 'orange', 'g', 'r', 'purple', 'c']``).
    - The output figure size is fixed to ``(7.5, 4)`` inches.
    """
	# Create plot space.
	fig, ax = plt.subplots(
		1, 1, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	# Add high-resolution coastlines.
	ax.coastlines(resolution='10m')

	# Fill land and ocean.
	ax.add_feature(cfeature.LAND, facecolor='#FFE6B3')

	# List of standard colors in Matplotlib
	colors = ['b', 'orange', 'g', 'r', 'purple', 'c']

	# Focus period.
	focus_period = iris.Constraint(
		time=lambda cell: focus_t0 <= cell.point <= focus_t1
	)

	# Loop through simulation tracks.
	for it in range(len(cubes)):
		# Calculate storm longitude and latitude.
		cube = cubes[it].extract(focus_period)
		storm_eowyn_lon, storm_eowyn_lat = calculate_crude_track(psl_cube=cube)
		
		# Plot result.
		ax.plot(storm_eowyn_lon, storm_eowyn_lat, label=labels[it], color=colors[it])

	# Add legend.
	plt.legend(loc=0, fontsize=6)

	# Add title.
	fig.suptitle(title, weight='bold')
	
	# Set size.
	fig.set_size_inches(7.5, 4)

	# Save.
	plt.savefig(fname)

	return True


def plot_tropical_cyclone_track(storm_name, focus_t0, focus_t1, cubes, title, xlim, ylim, legend, fname, buffer_deg=3):
	# Create plot space.
	fig, ax = plt.subplots(
		1, 1, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	# Add high-resolution coastlines.
	ax.coastlines(resolution='10m')

	# Set limits.
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	# List of standard colors in Matplotlib
	colors = ['b', 'orange', 'g', 'r', 'purple', 'c']

	# True storm track.
	storm_lon, storm_lat = get_true_track(storm_name, focus_t0, focus_t1)

	# Define track object for plot.
	storm_obj = sgeom.LineString(zip(storm_lon, storm_lat))
	# Add buffer for storm track.
	storm_buffer = storm_obj.buffer(1)
	# Add true storm track.
	storm_x, storm_y = storm_obj.xy
	ax.plot(storm_x, storm_y, label="Reference Track", color="k")
	# Add buffer.
	ax.add_geometries(
		[storm_buffer], ccrs.PlateCarree(), facecolor='#C8A2C8', alpha=0.5
	)

	# Focus period.
	focus_period = iris.Constraint(
		time=lambda cell: focus_t0 <= cell.point <= focus_t1
	)

	# Loop through simulation tracks.
	for it in range(len(cubes)):
		# Get grid of cube.
		lon, lat = cubes[it].coord("longitude").points, cubes[it].coord("latitude").points
		grid = np.meshgrid(lon, lat)

		# Determine crude storm track.
		sim_lon, sim_lat = calculate_crude_track(cubes[it], focus_period)

		# Define track object for plot.
		sim_obj = sgeom.LineString(zip(sim_lon, sim_lat))

		# Keep track if it is in the intersection of the buffer.
		sim_intersec = sim_obj.intersection(storm_obj.buffer(buffer_deg))

		# Get longest line if type is MultiLineString.
		if isinstance(sim_intersec, sgeom.MultiLineString):
			# Make iterable and choose longest LineString.
			sim_intersec = max(
				list(sim_intersec.geoms), key=lambda line: line.length
			)

		# Get longitude-latitude maximums and minimums.
		# Longitude.
		min_lon, max_lon = np.sort(lon)[4:].min(), np.sort(lon)[:-4].max()

		# Get simulation track.
		sim_x, sim_y = sim_intersec.xy
		# Convert to NumPy arrays.
		sim_x, sim_y = np.asarray(sim_x), np.asarray(sim_y)

		# Remove if on the bounary.
		bounds = np.logical_and(min_lon < sim_x, sim_x < max_lon)
		# Remomve from defined path.
		sim_x, sim_y = sim_x[bounds], sim_y[bounds]

		ax.plot(sim_x, sim_y, label=legend[it], color=colors[it])

	# Add legend.
	plt.legend(loc=0, fontsize=6)

	# Add title.
	fig.suptitle(title, weight='bold')
	
	# Set size.
	fig.set_size_inches(7.5, 4)

	# Save.
	plt.savefig(fname)

	return True

def create_weather_map_ani(pres_cubes, pr_cubes, focus_period, title, subtitles, unit, folder, size, interval):
	"""
    Create an animated weather map showing precipitation rate and mean sea-level pressure.

    For each timestep in the focus period, precipitation rate (shaded contours)
    is overlaid with mean sea-level pressure (black contour lines). The function
    generates both individual frame images (PNG) and an MP4 animation.

    Parameters
    ----------
    pres_cubes : list of list of iris.cube.Cube
        Nested list (2D structure) of sea-level pressure cubes. Each cube must
        include dimensions ``(time, latitude, longitude)`` and a ``time`` coordinate.
    pr_cubes : list of list of iris.cube.Cube
        Nested list (2D structure) of precipitation rate cubes, same shape as
        `pres_cubes`.
    focus_period : iris.Constraint
        Constraint selecting the time period to animate.
    title : str
        Overall title for the animation. The current date is appended automatically
        in each frame.
    subtitles : list of list of str
        Nested list of subplot titles, same shape as `pres_cubes` / `pr_cubes`.
    unit : str
        Unit string for the precipitation rate (e.g., ``"mm/hr"``). Used to
        label the colorbar.
    folder : str
        Path to the output folder. Individual frames (PNG) and the MP4 animation
        will be saved here. The folder is created if it does not exist.
    size : tuple of float
        Figure size in inches, e.g., (width, height).
    interval : int
        Delay between frames in milliseconds (passed to
        :class:`matplotlib.animation.FuncAnimation`).

    Returns
    -------
    success : bool
        Returns ``True`` after successfully creating and saving the animation.

    Notes
    -----
    - Precipitation shading uses :func:`prate_1hr_cmap` for contour levels and colormap.
    - Sea-level pressure contours are drawn every 5 hPa from 930 to 1040 hPa,
      with inline labels.
    - If ``"Reference: ERA5"`` is present in the title, only every fourth frame
      is saved as PNG; otherwise, all frames are saved.
    - A horizontal colorbar is added below the subplots, labelled with `unit`.
    - Coastlines are drawn with 10m resolution.
    - Progress is tracked with a :class:`tqdm` progress bar.
    - The final MP4 is saved as ``<folder>/<foldername>_ani.mp4`` where
      ``foldername`` is the basename of `folder`.
    """
	# Determine number of rows.
	nrows = len(pres_cubes)
	ncols = len(pres_cubes[0])

	# Create plot space.
	fig, axs = plt.subplots(
		nrows, ncols, dpi=300, subplot_kw={'projection': ccrs.PlateCarree()}
	)

	# Retrieve rainfall levels and create normalisation.
	# Retrieve colour map.
	cmap, levels = prate_1hr_cmap()
	# Create norm.
	norm = colors.BoundaryNorm(levels, cmap.N)
	# Add to list.
	cmapping = [levels, cmap, norm]

	# Loop through datasets.
	contourfs = []
	new_pres_cubes = []
	new_pr_cubes = []
	for i in range(nrows):
		pres_cubes_row = []
		pr_cubes_row = []
		for j in range(ncols):
			# Create longitude-latitude meshgrid.
			lon = pr_cubes[i][j].coord('longitude').points
			lat = pr_cubes[i][j].coord('latitude').points
			x, y = np.meshgrid(lon, lat)

			# Check if the grid is rectangular. 
			if lon.ndim == 1:
				# Generate mesh grid.
				x, y = np.meshgrid(lon, lat)
			else:
				# Get longitude-latitude as x-y.
				x, y = lon, lat

			# Get axis for plotting.
			if nrows == 1 and ncols == 1:
				ax = axs
			elif nrows == 1:
				ax = axs[j]
			elif ncols == 1:
				ax = axs[i]
			else:
				ax = axs[i][j]

			# Add high-resolution coastlines.
			ax.coastlines(resolution='10m')

			# Add subplot title.
			ax.set_title(subtitles[i][j])

			# Extract time period.
			# Pressure.
			pres_cube = pres_cubes[i][j].extract(focus_period)
			pres_cubes_row.append(pres_cube)
			# Rainfall.
			pr_cube = pr_cubes[i][j].extract(focus_period)
			pr_cubes_row.append(pr_cube)

			# Precipitation rate.
			contourf = ax.contourf(
				x, y, pr_cube[0].data, 
				levels=cmapping[0], cmap=cmapping[1], norm=cmapping[2], 
				extend="max"
			)
			contourfs.append(contourf)

		# Add row of time constrained cubes.
		new_pres_cubes.append(pres_cubes_row)
		new_pr_cubes.append(pr_cubes_row)

	# Redefine cubes variable.
	pres_cubes = new_pres_cubes
	pr_cubes = new_pr_cubes
	
	# Update function for animation.
	def update(frame, pres_cubes, pr_cubes, title, levels, pbar, folder, cmapping):
		# Determine number of rows.
		nrows = len(pr_cubes)
		ncols = len(pr_cubes[0])

		# Remove plots.
		# Define global.
		global contours
		global contourfs
		# Try removal.
		try:
			for contourf in contourfs:
				contourf.remove()
			for contour in contours:
				contour.remove()
		except:
			pass

		# Determine date.
		cube_time = pr_cubes[0][0].coord("time")
		date = cube_time.units.num2date(cube_time.points)[frame]

		# Loop through cubes.
		contours = []
		contourfs = []
		for i in range(nrows):
			for j in range(ncols):
				# Create longitude-latitude meshgrid.
				lon = pr_cubes[i][j].coord('longitude').points
				lat = pr_cubes[i][j].coord('latitude').points

				# Check if the grid is rectangular. 
				if lon.ndim == 1:
					# Generate mesh grid.
					x, y = np.meshgrid(lon, lat)
				else:
					# Get longitude-latitude as x-y.
					x, y = lon, lat

				# Get relevant cube.
				pres_cube_data = pres_cubes[i][j][frame].data
				pr_cube_data = pr_cubes[i][j][frame].data

				# Get axis for plotting.
				if nrows == 1 and ncols == 1:
					ax = axs
				elif nrows == 1:
					ax = axs[j]
				elif ncols == 1:
					ax = axs[i]
				else:
					ax = axs[i][j]

				# Precipitation rate.
				contourf = ax.contourf(
					x, y, pr_cube_data, 
					levels=cmapping[0], cmap=cmapping[1], norm=cmapping[2], 
					extend="max"
				)

				# Mean sea level pressure.
				pres_levels = np.arange(930, 1040 + 5, 5)
				contour = ax.contour(
					x, y, pres_cube_data, levels=pres_levels, linewidths=0.5, colors='k'
				)

				# Add labels.
				ax.clabel(contour, inline=True, fontsize=8)

				# Append to list.
				contours.append(contour)

				# Append contourf.	
				contourfs.append(contourf)

		# Add title.
		fig.suptitle(
			"{}\nDate: {}".format(title, date.strftime('%d-%m-%Y %H:%M')), 
			weight='bold'
		)

		# Save frame.
		fname = "{}_{}.png".format(
			folder.rsplit('/', 1)[-1], date.strftime('%Y%m%d%H%M')
		)

		if not "Reference: ERA5" in title:
			# Save every frame is gridded dataset.
			plt.savefig("{}/{}".format(folder, fname))
		else:
			# Save every fourth frame if ERA5.
			if frame % 4 == 0:
				plt.savefig("{}/{}".format(folder, fname))

		# Update progress bar.
		pbar.update(1)

		return contourfs

	# Initialize tqdm progress bar
	pbar = tqdm(
		total=pr_cubes[i][j].shape[0], desc="Animating parameter plots"
	)

	# Create animation.
	ani = animation.FuncAnimation(
		fig, 
		update, 
		frames=pr_cubes[i][j].shape[0], 
		fargs=(pres_cubes, pr_cubes, title, levels, pbar, folder, cmapping),
		interval=interval
	)

	# Create colour bar.
	cbar = plt.colorbar(
		contourf, 
		ax=fig.axes, 
		aspect=60, 
		orientation="horizontal", 
		extend="both", 
		fraction=0.02
	)
	cbar.set_label(unit)
	cbar.ax.tick_params(length=0)

	# Set size.
	fig.set_size_inches(size)

	# Check if folder exists.
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Save file as MP4.
	fname = "{}_ani.mp4".format(folder.rsplit('/', 1)[-1])
	ani.save(filename="{}/{}".format(folder, fname))

	# Close progress bar.
	pbar.close()

	return True

# List of all functions for documentation.
__all__ = [
    "load_grib_files",
    "load_hclim_netcdf",
    "create_rectilinear_cube",
    "regrid_cube",
    "extract_common_period",
    "calculate_rainfall_accumulation",
    "add_statistical_information",
    "plot_dist",
    "plot_series",
    "plot_diurnal_cycle",
    "find_nearest_grid_point",
    "plot_location_series",
    "create_param_plot",
    "rainfall_cmap",
    "prate_1hr_cmap",
    "create_bias_plot",
    "plot_extratropical_cyclone_track",
    "calculate_crude_track",
    "create_param_ani",
    "create_bias_ani",
    "create_weather_map_ani",
]
