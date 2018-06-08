import abc
from enum import Enum
import numpy as np


class GridType(Enum):
    """
    Enumeration of grid types
    """
    structured = 0
    layered_vertex = 1
    unlayered_vertex = 2


class LocationType(Enum):
    """
    Enumeration of location types
    """
    xyz = 0  # coordinates defined by spatial reference
    xyz_nosr = 1 # coordinates defined based on 0,0 point as upper-left most
                 # point of the model
    cellid = 2  # cell ids
    layer_cellid = 3  # layer + cell id
    lrc = 4  # layer row column


class TimeType(Enum):
    """
    Enumeration of time types
    """
    calendar = 0 # year/month/day/hour/minute/second
    sp = 1 # stress period
    sp_ts = 2 # stress period time step


class PointType(Enum):
    """
    Enumeration of vertex types
    """
    modelxy = 0
    spatialxyz = 1


class MFGridException(Exception):
    """
    Model grid related exception
    """
    def __init__(self, error):
        Exception.__init__(self, "MFGridException: {}".format(error))


class SimulationTime():
    """
    Class for MODFLOW simulation time

    Parameters
    ----------
    stress_periods : pandas dataframe
        headings are: perlen, nstp, tsmult
    temporal_reference : TemporalReference
        contains start time and time units information
    """
    def __init__(self, stress_periods, temporal_reference=None):
        self.stress_periods = stress_periods
        self.temporal_reference = temporal_reference


class CachedData():
    def __init__(self, data):
        self.data = data
        self.out_of_date = False

    def update_data(self, data):
        self.data = data
        self.out_of_date = False


class ModelGrid(object):
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    grid_type : enumeration
        type of model grid (DiscritizationType.DIS, DiscritizationType.DISV,
        DiscritizationType.DISU)
    sr : SpatialReference
        Spatial reference locates the grid in a coordinate system
    simulation_time : SimulationTime
        Simulation time provides time information for temporal grid data
    model_name : str
        Name of the model associated with this grid

    Methods
    ----------
    get_tabular_data : (name_list, data, location_type, time_type) : data
        returns a pandas object with the data defined in name_list in the
        spatial representation defined by coord_type and using time
        representation defined by time_type.
    get_cell_centers : () : ndarray
        returns the cell centers of all models cells in the model grid.  if
        the model grid contains spatial reference information, the cell centers
        are in the coordinate system provided by the spatial reference
        information. otherwise the cell centers are based on a 0,0 location for
        the upper left corner of the model grid
    vertices : (point_type) : ndarray
        1D array of cell vertices for whole grid in C-style (row-major) order
        (same as np.ravel())
    get_model_dim : () : list
        returns the dimensions of the model
    get_horizontal_cross_section_dim_names : () : list
        returns the appropriate dimension axis for a horizontal cross section
        based on the model discritization type
    get_model_dim_names : () : list
        returns the names of the model dimensions based on the model
        discritization type
    get_num_spatial_coordinates : () : int
        returns the number of spatial coordinates
    num_connections : () : int
        returns the number of model connections.  model discritization type
        must be DIS
    num_cells_per_layer : () : list
        returns the number of cells per model layer.  model discritization
        type must be DIS or DISV
    num_layers : () : int
        returns the number of layers in the model, if the model is layered
    num_cells : () : int
        returns the total number of cells in the model
    get_all_model_cells : () : list
        returns a list of all model cells, represented as a layer/row/column
        tuple, a layer/cellid tuple, or a cellid for the structured, layered
        vertex, and vertex grids respectively

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, grid_type, sr=None, simulation_time=None,
                 model_name=''):
        self.grid_type = grid_type
        self._sr = sr
        self.sim_time = simulation_time
        self.model_name = model_name
        self._cache_dict = {}

    ###########################
    # basic functions
    ###########################
    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr):
        self._sr = sr
        self._require_cache_updates()

    @abc.abstractmethod
    def get_tabular_data(self, data, coord_type=LocationType.xyz,
                        time_type=TimeType.calendar):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    def _require_cache_updates(self):
        for cache_data in self._cache_dict.values():
            cache_data.out_of_date = True

    ############################
    # from spatial reference
    ############################
    @property
    # it would be nice if all of these worked with or without a spatial
    # reference, and the user has the options of getting the output in the
    # spatial reference system or just as the distance from a corner
    # of the model (say upper-left corner)
    def xedge(self):
        return self.get_edge_array()[0]

    @property
    def yedge(self):
        return self.get_edge_array()[1]

    @property
    def xgrid(self):
        return self.get_xygrid[0]

    @property
    def ygrid(self):
        return self.get_xygrid[1]

    def xcell_centers(self, point_type=PointType.xyz):
        return self.get_cellcenters()[0]

    def ycell_centers(self, point_type=PointType.xyz):
        return self.get_cellcenters()[1]

    @abc.abstractmethod
    def vertices(self, point_type=PointType.xyz):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    def get_cellcenters(self):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    def get_xygrid(self):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')
        #xgrid, ygrid = np.meshgrid(self.xedge, self.yedge)
        #return self.transform(xgrid, ygrid)

    @abc.abstractmethod
    def get_edge_array(self):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    def write_gridSpec(self, filename):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    def get_quadmesh_plotter(self):
        """
        Create a QuadMesh plotting object for this grid

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    def plot_array(self, a, ax=None, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        a : np.ndarray

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move export-specific code to export
    def export_array(self, filename, a, nodata=-9999,
                     fieldname='value',
                     **kwargs):
        """Write a numpy array to Arc Ascii grid
        or shapefile with the model reference.

        Parameters
        ----------
        filename : str
            Path of output file. Export format is determined by
            file extention.
            '.asc'  Arc Ascii grid
            '.tif'  GeoTIFF (requries rasterio package)
            '.shp'  Shapefile
        a : 2D numpy.ndarray
            Array to export
        nodata : scalar
            Value to assign to np.nan entries (default -9999)
        fieldname : str
            Attribute field name for array values (shapefile export only).
            (default 'values')
        kwargs:
            keyword arguments to np.savetxt (ascii)
            rasterio.open (GeoTIFF)
            or flopy.export.shapefile_utils.write_grid_shapefile2

        Notes
        -----
        Rotated grids will be either be unrotated prior to export,
        using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
        included in their transform property (GeoTiff format). In either case
        the pixels will be displayed in the (unrotated) projected geographic coordinate system,
        so the pixels will no longer align exactly with the model grid
        (as displayed from a shapefile, for example). A key difference between
        Arc Ascii and GeoTiff (besides disk usage) is that the
        unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
        will have the same number of rows and pixels as the original.
        """
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move export-specific code to export
    def export_contours(self, filename, contours,
                        fieldname='level', epsg=None, prj=None,
                        **kwargs):
        """Convert matplotlib contour plot object to shapefile.

        Parameters
        ----------
        filename : str
            path of output shapefile
        contours : matplotlib.contour.QuadContourSet or list of them
            (object returned by matplotlib.pyplot.contour)
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp

        Returns
        -------
        df : dataframe of shapefile contents
        """
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move export-specific code to export
    def export_array_contours(self, filename, a,
                              fieldname='level',
                              interval=None,
                              levels=None,
                              maxlevels=1000,
                              epsg=None,
                              prj=None,
                              **kwargs):
        """Contour an array using matplotlib; write shapefile of contours.

        Parameters
        ----------
        filename : str
            Path of output file with '.shp' extention.
        a : 2D numpy array
            Array to contour
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp
        """
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move plot-specific code
    def contour_array(self, ax, a, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            ax to add the contours

        a : np.ndarray
            array to contour

        Returns
        -------
        contour_set : ContourSet

        """
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    # more specific name for this?
    @abc.abstractmethod
    def interpolate(self, a, xi, method='nearest'):
        """
        Use the griddata method to interpolate values from an array onto the
        points defined in xi.  For any values outside of the grid, use
        'nearest' to find a value for them.

        Parameters
        ----------
        a : numpy.ndarray
            array to interpolate from.  It must be of size nrow, ncol
        xi : numpy.ndarray
            array containing x and y point coordinates of size (npts, 2). xi
            also works with broadcasting so that if a is a 2d array, then
            xi can be passed in as (xgrid, ygrid).
        method : {'linear', 'nearest', 'cubic'}
            method to use for interpolation (default is 'nearest')

        Returns
        -------
        b : numpy.ndarray
            array of size (npts)

        """
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_2d_vertex_connectivity(self):
        """
        Create the cell 2d vertices array and the iverts index array.  These
        are the same form as the ones used to instantiate an unstructured
        spatial reference.

        Returns
        -------

        verts : ndarray
            array of x and y coordinates for the grid vertices

        iverts : list
            a list with a list of vertex indices for each cell in clockwise
            order starting with the upper left corner

        """
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    def get_rc(self, x, y):
        """Return the row and column of a point or sequence of points
        in real-world coordinates.

        Parameters
        ----------
        x : scalar or sequence of x coordinates
        y : scalar or sequence of y coordinates

        Returns
        -------
        r : row or sequence of rows (zero-based)
        c : column or sequence of columns (zero-based)
        """
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    # what is the difference between this and the one without "shared" below?
    def get_3d_shared_vertex_connectivity(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_3d_vertex_connectivity(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    #################################
    # from flopy for mf6 model grid
    #################################
    @abc.abstractmethod
    def get_row_array(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_column_array(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_layer_array(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_arraydata(self, name):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_model_dim(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_model_dim_names(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_horizontal_cross_section_dim_names(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_horizontal_cross_section_dim_arrays(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_model_dim_arrays(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def num_connections(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def num_cells_per_layer(self, active_only=False):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def num_cells(self, active_only=False):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_all_model_cells(self, active_only=False):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    # move to export folder
    def load_shapefile_data(self, shapefile):
        # this will only support loading shapefiles with the same
        # spatial reference as the model grid
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')


class StructuredModelGrid(ModelGrid):
    """
    get_row_array : ()
        returns a numpy ndarray sized to a model row
    get_column_array : ()
        returns a numpy ndarray sized to a model column
    get_layer_array : ()
        returns a numpy ndarray sized to a model layer
    num_rows
        returns the number of model rows
    num_columns
        returns the number of model columns

    """
    def __init__(self, delc, delr, top, botm, idomain, sr=None,
                 simulation_time=None):
        super(StructuredModelGrid, self).__init__(GridType.structured, sr,
                                                  simulation_time)
        self.delc = delc
        self.delr = delr
        self.top = top
        self.botm = botm
        self.idomain = idomain
        self.nlay = len(botm)
        self.nrow = len(delr)
        self.ncol = len(delc)

    def get_arraydata(self, name):
        raise NotImplementedError('this feature is not yet implemented')

    def num_connections(self):
        raise NotImplementedError('this feature is not yet implemented')

    def get_all_model_cells(self, active_only=False):
        raise NotImplementedError('this feature is not yet implemented')

    # move to export folder
    def load_shapefile_data(self, shapefile):
        # this will only support loading shapefiles with the same
        # spatial reference as the model grid
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    def get_row_array(self):
        return np.arange(1, self.nrow + 1, 1, np.int)

    def get_column_array(self):
        return np.arange(1, self.ncol + 1, 1, np.int)

    def get_layer_array(self):
        return np.arange(1, self.nlay + 1, 1, np.int)

    def get_model_dim_arrays(self):
        return [np.arange(1, self.nlay + 1, 1, np.int),
                np.arange(1, self.nrow + 1, 1, np.int),
                np.arange(1, self.ncol + 1, 1, np.int)]

    def num_cells_per_layer(self, active_only=False):
        if active_only:
            raise NotImplementedError('this feature is not yet implemented')
        else:
            return self.nrow * self.ncol

    def num_cells(self, active_only=False):
        if active_only:
            raise NotImplementedError('this feature is not yet implemented')
        else:
            return self.nrow * self.ncol * self.nlay

    def get_model_dim(self):
        if self.grid_type() == GridType.structured:
            return [self.nlay, self.nrow, self.ncol]
        elif self.grid_type() == GridType.layered_vertex:
            return [self.nlay, self.num_cells_per_layer()]
        elif self.grid_type() == GridType.unlayered_vertex:
            return [self.num_cells()]

    def get_model_dim_names(self):
        if self.grid_type() == GridType.structured:
            return ['layer', 'row', 'column']
        elif self.grid_type() == GridType.layered_vertex:
            return ['layer', 'layer_cell_num']
        elif self.grid_type() == GridType.unlayered_vertex:
            return ['node']

    def get_horizontal_cross_section_dim_names(self):
        return ['row', 'column']

    def get_horizontal_cross_section_dim_arrays(self):
        return [np.arange(1, self.nrow + 1, 1, np.int),
                np.arange(1, self.ncol + 1, 1, np.int)]


class VertexModelGrid(ModelGrid):
    def __init__(self, top, botm, idomain, vertices, cell2d, nlay=None,
                 ncpl=None, sr=None, simulation_time=None):
        if nlay is None:
            grid_type = GridType.unlayered_vertex
        else:
            grid_type = GridType.layered_vertex
        super(VertexModelGrid, self).__init__(grid_type, sr, simulation_time)
        self.top = top
        self.botm = botm
        self.idomain = idomain
        self._vertices = vertices
        self.cell2d = cell2d
        self.nlay = nlay
        self.ncpl = ncpl

    def get_model_dim_arrays(self):
        if self.grid_type() == GridType.layered_vertex:
            return [np.arange(1, self.nlay + 1, 1, np.int),
                    np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == GridType.unlayered_vertex:
            return [np.arange(1, self.num_cells() + 1, 1, np.int)]

    def num_cells_per_layer(self, active_only=False):
        if self.grid_type() == GridType.layered_vertex:
            # return number of cells for each layer
            raise NotImplementedError(
                'this feature is not yet implemented')
        elif self.grid_type() == GridType.unlayered_vertex:
            except_str = 'ERROR: Model "{}" is unstructured and does not ' \
                         'have a consistant number of cells per ' \
                         'layer.'.format(self.model_name)
            print(except_str)
            raise MFGridException(except_str)

    def num_cells(self, active_only=False):
        if active_only:
            raise NotImplementedError(
                'this feature is not yet implemented')
        else:
            if self.grid_type() == GridType.layered_vertex:
                total_cells = 0
                for layer_cells in self.ncpl:
                    total_cells += layer_cells
                return total_cells
            elif self.grid_type() == GridType.unlayered_vertex:
                raise NotImplementedError(
                    'this feature is not yet implemented')

    def get_model_dim(self):
        if self.grid_type() == GridType.layered_vertex:
            return [self.nlay, max(self.ncpl)]
        elif self.grid_type() == GridType.unlayered_vertex:
            return [self.num_cells()]

    def get_model_dim_names(self):
        if self.grid_type() == GridType.structured:
            return ['layer', 'row', 'column']
        elif self.grid_type() == GridType.layered_vertex:
            return ['layer', 'layer_cell_num']
        elif self.grid_type() == GridType.unlayered_vertex:
            return ['node']

    def get_horizontal_cross_section_dim_names(self):
        if self.grid_type() == GridType.layered_vertex:
            return ['layer_cell_num']
        elif self.grid_type() == GridType.unlayered_vertex:
            except_str = 'ERROR: Can not get layer dimension name for model ' \
                         '"{}" DISU grid. DISU grids do not support ' \
                         'layers.'.format(self.model_name)
            print(except_str)
            raise MFGridException(except_str)

    def get_horizontal_cross_section_dim_arrays(self):
        if self.grid_type() == GridType.layered_vertex:
            return [np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == GridType.unlayered_vertex:
            except_str = 'ERROR: Can not get horizontal plane arrays for ' \
                         'model "{}" DISU grid.  DISU grids do not support ' \
                         'individual layers.'.format(self.model_name)
            print(except_str)
            raise MFGridException(except_str)

    @classmethod
    # move to export folder
    def from_argus_export(cls, fname, nlay=1):
        """
        Create a new SpatialReferenceUnstructured grid from an Argus One
        Trimesh file

        Parameters
        ----------
        fname : string
            File name

        nlay : int
            Number of layers to create

        Returns
        -------
            sru : flopy.utils.reference.SpatialReferenceUnstructured

        """
        from ..utils.geometry import get_polygon_centroid
        f = open(fname, 'r')
        line = f.readline()
        ll = line.split()
        ncells, nverts = ll[0:2]
        ncells = int(ncells)
        nverts = int(nverts)
        verts = np.empty((nverts, 2), dtype=np.float)
        xc = np.empty((ncells), dtype=np.float)
        yc = np.empty((ncells), dtype=np.float)

        # read the vertices
        f.readline()
        for ivert in range(nverts):
            line = f.readline()
            ll = line.split()
            c, iv, x, y = ll[0:4]
            verts[ivert, 0] = x
            verts[ivert, 1] = y

        # read the cell information and create iverts, xc, and yc
        iverts = []
        for icell in range(ncells):
            line = f.readline()
            ll = line.split()
            ivlist = []
            for ic in ll[2:5]:
                ivlist.append(int(ic) - 1)
            if ivlist[0] != ivlist[-1]:
                ivlist.append(ivlist[0])
            iverts.append(ivlist)
            xc[icell], yc[icell] = get_polygon_centroid(verts[ivlist, :])

        # close file and return spatial reference
        f.close()
        return cls(xc, yc, verts, iverts, np.array(nlay * [len(iverts)]))
