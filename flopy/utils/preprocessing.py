import os
import time
import collections
import numpy as np
import pandas as pd
from flopy.grid.reference import getproj4
from flopy.export.shapefile_utils import shp2recarray

def get_values_at_points(rasterfile, x=None, y=None,
                         points=None):
    """Get raster values single point or list of points.
    Points must be in same coordinate system as raster.

    Parameters
    ----------
    rasterfile : str
        Filename of raster.
    x : 1D array
        X coordinate locations
    y : 1D array
        Y coordinate locations
    points : list of tuples or 2D numpy array (npoints, (row, col))
        Points at which to sample raster.

    Returns
    -------
    list of floats

    Notes
    -----
    requires gdal
    """
    try:
        import gdal
    except:
        print('This function requires gdal.')

    if x is not None and isinstance(x[0], tuple):
        x, y = np.array(x).transpose()
        warnings.warn(
            "new argument input for get_values_at_points is x, y, or points"
        )
    elif x is not None:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.ndarray(y)
    elif points is not None:
        if not isinstance(points, np.ndarray):
            x, y = np.array(points)
        else:
            x, y = points[:, 0], points[:, 1]
    else:
        print('Must supply x, y or list/array of points.')

    assert os.path.exists(rasterfile), "raster {} not found".format(rasterfile)
    t0 = time.time()
    # open the raster
    gdata = gdal.Open(rasterfile)

    # get the location info
    xul, dx, rx, yul, ry, dy = gdata.GetGeoTransform()

    # read the array
    print("reading data from {}...".format(rasterfile))
    data = gdata.ReadAsArray().astype(np.float)
    nrow, ncol = data.shape

    print("sampling...")
    # find the closest row, col location for each point
    i = ((y - yul) / dy).astype(int)
    j = ((x - xul) / dx).astype(int)

    # mask row, col locations outside the raster
    within = (i > 0) & (i < nrow) & (j > 0) & (j < ncol)

    # get values at valid point locations
    results = np.ones(len(i), dtype=float) * np.nan
    results[within] = data[i[within], j[within]]

    print("finished in {:.2f}s".format(time.time() - t0))
    return results

def intersect(feature, sr, id_column=None,
              epsg=None,
              proj4=None):
    """Intersect a feature with the model grid, using
    the rasterio.features.rasterize method. Features are intersected
    if they contain the cell center.

    Parameters
    ----------
    feature : str (shapefile path), list of shapely objects,
              or dataframe with geometry column
    id_column : str
        Column with unique integer identifying each feature; values
        from this column will be assigned to the output raster.
    sr : flopy.utils.SpatialReference instance
    epsg : int
        EPSG code for feature coordinate reference system. Optional,
        but an epgs code or proj4 string must be supplied if feature
        isn't in the same CRS as the model.
    proj4 : str
        Proj4 string for feature CRS (optional)

    Returns
    -------
    2D numpy array with intersected values

    """
    try:
        from rasterio import features
        from rasterio import Affine
    except:
        print('This method requires rasterio. Try conda install rasterio.')
        return

    trans = Affine(sr.delr[0]*sr.length_multiplier, 0., sr.xul,
                   0., -sr.delc[0]*sr.length_multiplier, sr.yul) * Affine.rotation(sr.rotation)

    if isinstance(feature, str):
        df = pd.DataFrame(shp2recarray(feature))
    elif not isinstance(feature, collections.Iterable):
        df = pd.DataFrame({'geometry': [feature]})
    elif isinstance(feature, pd.DataFrame):
        df = feature.copy()
    else:
        print('unrecognized feature input')
        return

    # handle shapefiles in different CRS than model grid
    reproject = False
    if proj4 is not None:
        if proj4 != sr.proj4_str:
            reproject = True
    elif epsg is not None and sr.epsg is not None:
        if epsg != sr.epsg:
            reproject = True
            proj4 = getproj4(epsg)
    if reproject:
        print('Reprojection from other CRS not implemented yet.')

    # create list of GeoJSON features, with unique value for each feature
    if id_column is None:
        numbers = range(1, len(df)+1)
    else:
        numbers = df[id_column].tolist()
    geojson = [g.geojson for g in df.geometry]
    geoms = list(zip(geojson, numbers))
    result = features.rasterize(geoms,
                                out_shape=(sr.nrow, sr.ncol),
                                transform=trans)
    return result.astype(np.int32)