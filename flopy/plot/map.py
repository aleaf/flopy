import copy
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    plt = None

from . import plotutil
# from flopy.plot.plotutil import bc_color_dict
from ..utils import SpatialReference as DepreciatedSpatialReference
from ..grid.structuredmodelgrid import StructuredModelGrid
from ..grid.vertexmodelgrid import VertexModelGrid
from ..grid.reference import SpatialReference
import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class StructuredMapView(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    sr : flopy.utils.reference.SpatialReference
        The spatial reference class (Default is None)
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy.modflow.ModflowDis object
        flopy discretization object. (Default is None)
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner.  The default is the sum of the
        delc array.
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    Notes
    -----
    ModelMap must know the position and rotation of the grid in order to make
    the plot.  This information is contained in the SpatialReference class
    (sr), which can be passed.  If sr is None, then it looks for sr in dis.
    If dis is None, then it looks for sr in model.dis.  If all of these
    arguments are none, then it uses xul, yul, and rotation.  If none of these
    arguments are provided, then it puts the lower-left-hand corner of the
    grid at (0, 0).

    """

    def __init__(self, sr=None, ax=None, model=None, dis=None, modelgrid=None,
                 layer=0, extent=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., length_multiplier=1.):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise Exception(s)

        self.model = model
        self.layer = layer
        self.dis = dis
        self.mg = None
        self.sr = None

        if model is not None:
            self.mg = copy.deepcopy(model.modelgrid)
            self.sr = copy.deepcopy(model.modelgrid.sr)

        elif modelgrid is not None:
            self.mg = copy.deepcopy(modelgrid)
            self.sr = copy.deepcopy(modelgrid.sr)

        elif dis is not None:
            self.mg = copy.deepcopy(dis.parent.modelgrid)
            self.sr = copy.deepcopy(dis.parent.modelgrid.sr)

        elif sr is not None:
            if isinstance(sr, DepreciatedSpatialReference):
                self.mg = copy.deepcopy(sr)
                self.sr = copy.deepcopy(sr)

            else:
                self.sr = sr
                self.mg = StructuredModelGrid(delc=np.array([]), delr=np.array([]),
                                              top=np.array([]), botm=np.array([]),
                                              idomain=np.array([]), sr=self.sr)

        else:
            self.sr = SpatialReference(delc=np.array([]), xll=xll, xul=xul,
                                       yul=yul, rotation=rotation,
                                       length_multiplier=length_multiplier)
            self.mg = StructuredModelGrid(delc=np.array([]), delr=np.array([]),
                                          top=np.array([]), botm=np.array([]),
                                          idomain=np.array([]), sr=self.sr)

        # model map override spatial reference settings
        if any(elem is not None for elem in (xul, yul, xll, yll)) or \
                rotation != 0 or length_multiplier != 1.:
            self.sr.length_multiplier = length_multiplier
            if isinstance(sr, DepreciatedSpatialReference):
                self.sr.set_spatialreference(xul=xul, yul=yul,
                                             xll=xll, yll=yll,
                                             rotation=rotation)
            else:
                self.sr.set_spatialreference(delc=self.mg.delc,
                                             xul=xul, yul=yul,
                                             xll=xll, yll=yll,
                                             rotation=rotation)
                self.mg.sr = self.sr

        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect('equal')
            except:
                self.ax = plt.subplot(1, 1, 1, aspect='equal', axisbg="white")
        else:
            self.ax = ax

        if extent is not None:
            self._extent = extent
        else:
            self._extent = None

    @property
    def extent(self):
        if self._extent is None:
            self._extent = self.mg.get_extent()
        return self._extent

    def plot_array(self, a, masked_values=None, **kwargs):
        """
        Plot an array.  If the array is three-dimensional, then the method
        will plot the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2 or 3')

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        try:
            # check if this is an old style spatial reference
            xgrid = self.sr.xgrid
            ygrid = self.sr.ygrid
        except AttributeError:
            xgrid, ygrid = self.mg.get_xygrid()

        quadmesh = ax.pcolormesh(xgrid, ygrid, plotarray)

        # set max and min
        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None
        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None

        quadmesh.set_clim(vmin=vmin, vmax=vmax)

        # send rest of kwargs to quadmesh
        quadmesh.set(**kwargs)

        # add collection to axis
        ax.add_collection(quadmesh)

        # set limits
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        return quadmesh

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        try:
            import matplotlib.tri as tri
        except:
            tri = None

        try:
            xcentergrid = self.mg.xcell_centers()
            ycentergrid = self.mg.ycell_centers()
        except AttributeError:
            xcentergrid = self.sr.xcentergrid
            ycentergrid = self.sr.ycentergrid

        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2 or 3')

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' in kwargs.keys():
            if 'cmap' in kwargs.keys():
                kwargs.pop('cmap')

        plot_triplot = False
        if 'plot_triplot' in kwargs:
            plot_triplot = kwargs.pop('plot_triplot')

        if 'extent' in kwargs and tri is not None:
            extent = kwargs.pop('extent')


            idx = (xcentergrid >= extent[0]) & (
                   xcentergrid <= extent[1]) & (
                          ycentergrid >= extent[2]) & (
                          ycentergrid <= extent[3])
            a = a[idx].flatten()
            xc = xcentergrid[idx].flatten()
            yc = ycentergrid[idx].flatten()
            triang = tri.Triangulation(xc, yc)
            try:
                amask = a.mask
                mask = [False for i in range(triang.triangles.shape[0])]
                for ipos, (n0, n1, n2) in enumerate(triang.triangles):
                    if amask[n0] or amask[n1] or amask[n2]:
                        mask[ipos] = True
                triang.set_mask(mask)
            except:
                mask = None
            contour_set = ax.tricontour(triang, plotarray, **kwargs)
            if plot_triplot:
                ax.triplot(triang, color='black', marker='o', lw=0.75)
        else:

            contour_set = ax.contour(xcentergrid, ycentergrid,
                                     plotarray, **kwargs)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self, ibound=None, color_noflow='black', **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound
        from the self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)

        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if ibound is None:
            bas = self.model.get_package('BAS6')
            ibound = bas.ibound.array

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return quadmesh

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if ibound is None:
            bas = self.model.get_package('BAS6')
            ibound = bas.ibound.array
        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
        return quadmesh

    def plot_grid(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
        kwargs : ax, colors.  The remaining kwargs are passed into the
            the LineCollection constructor.

        Returns
        -------
        lc : matplotlib.collections.LineCollection
p
        """
        from matplotlib.collections import LineCollection

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' not in kwargs:
            kwargs['colors'] = '0.5'

        lc = LineCollection(self.mg.get_grid_lines(), **kwargs)
        ax.add_collection(lc)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return lc

    def plot_bc(self, ftype=None, package=None, kper=0, color=None,
                plotAll=False, **kwargs):
        """
        Plot boundary conditions locations for a specific boundary
        type from a flopy model

        Parameters
        ----------
        ftype : string
            Package name string ('WEL', 'GHB', etc.). (Default is None)
        package : flopy.modflow.Modflow package class instance
            flopy package class instance. (Default is None)
        kper : int
            Stress period to plot
        color : string
            matplotlib color string. (Default is None)
        plotAll : bool
            Boolean used to specify that boundary condition locations for all
            layers will be plotted on the current ModelMap layer.
            (Default is False)
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        # Find package to plot
        if package is not None:
            p = package
            ftype = p.name[0]

        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            ftype = ftype.upper()
            p = self.model.get_package(ftype)

        else:
            raise Exception('Cannot find package to plot')

        # Get the list data

        # try:
        #    mflist = p.stress_period_data.data[kper]
        #except AttributeError:
            # todo: sanity check on this fp6 code!
            # todo: kper doesn't work, doesn't seem to return all data!
        #    mflist = p.stress_period_data.array # [kper]
        # else:
        #    raise Exception('Not a list-style boundary package:')

        # use a general expression to get stress period data
        arr_dict = p.stress_period_data.to_array(kper)
        if not arr_dict:
            return None

        for key in arr_dict:
            fluxes = arr_dict[key]
            break

        # Return if MfList is None
        # if mflist is None:
        #    return None
        nlay = self.model.modelgrid.nlay

        # Plot the list locations
        plotarray = np.zeros((nlay, self.mg.nrow, self.mg.ncol), dtype=np.int)
        if plotAll:
            #try:
            #    idx = [mflist['i'], mflist['j']]
            #except ValueError:
            #    idx = [[v[1] for v in mflist['cellid']],
            #           [v[2] for v in mflist['cellid']]]
            # plotarray[:, idx] = 1
            t = np.sum(fluxes, axis=0)
            pa = np.zeros((self.mg.nrow, self.mg.ncol), dtype=np.int)
            pa[t != 0] = 1
            for k in range(nlay):
                plotarray[k, :, :] = pa.copy()
        else:
            # try:
            #     idx = [mflist['k'], mflist['i'], mflist['j']]
            # except ValueError:
            # todo: sanity check on this fp6 code
            #     k = [v[0] for v in mflist['cellid']]
            #     i = [v[1] for v in mflist['cellid']]
            #     j = [v[2] for v in mflist['cellid']]
            #     idx = [k, i, j]

            plotarray[fluxes != 0] = 1

        # mask the plot array
        plotarray = np.ma.masked_equal(plotarray, 0)

        # set the colormap
        if color is None:
            if ftype in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[ftype]
            else:
                c = plotutil.bc_color_dict['default']
        else:
            c = color

        cmap = matplotlib.colors.ListedColormap(['0', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # create normalized quadmesh
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return quadmesh

    def plot_shapefile(self, shp, **kwargs):
        """
        Plot a shapefile.  The shapefile must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        shp : string
            Name of the shapefile to plot

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_shapefile()

        """
        err_msg = "plot_shapefile() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def plot_cvfd(self, verts, iverts, **kwargs):
        """
        Plot a cvfd grid.  The vertices must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        verts : ndarray
            2d array of x and y points.
        iverts : list of lists
            should be of len(ncells) with a list of vertex number for each cell

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_cvfd()

        """
        err_msg = "plot_cvfd() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def contour_array_cvfd(self, vertc, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        vertc : np.ndarray
            Array with of size (nc, 2) with centroid location of cvfd
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        err_msg = "contour_array_cvfd() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def plot_discharge(self, frf, fff, dis=None, flf=None, head=None, istep=1,
                       jstep=1, normalize=False, **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        frf : numpy.ndarray
            MODFLOW's 'flow right face'
        fff : numpy.ndarray
            MODFLOW's 'flow front face'
        dis: flopy.modflow.ModflowDis
            Flopy DIS package class
        flf : numpy.ndarray
            MODFLOW's 'flow lower face' (Default is None.)
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        istep : int
            row frequency to plot. (Default is 1.)
        jstep : int
            column frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors of specific discharge.

        """
        # remove 'pivot' keyword argument
        # by default the center of the arrow is plotted in the center of a cell
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        # Calculate specific discharge
        # make sure dis is defined
        if dis is None:
            if self.model is not None:
                dis = self.model.dis
            else:
                print('ModelMap.plot_quiver() error: self.dis is None and dis '
                      'arg is None.')
                return
        # todo: this will break with flopy6 it will have to call idomain array
        ib = self.model.bas6.ibound.array
        delr = dis.delr.array
        delc = dis.delc.array
        top = dis.top.array
        botm = dis.botm.array
        nlay, nrow, ncol = botm.shape
        laytyp = None
        hnoflo = 999.
        hdry = 999.
        if self.model is not None:
            lpf = self.model.get_package('LPF')
            if lpf is not None:
                laytyp = lpf.laytyp.array
                hdry = lpf.hdry
            bas = self.model.get_package('BAS6')
            if bas is not None:
                hnoflo = bas.hnoflo

        # If no access to head or laytyp, then calculate confined saturated
        # thickness by setting laytyp to zeros
        if head is None or laytyp is None:
            head = np.zeros(botm.shape, np.float32)
            laytyp = np.zeros((nlay,), dtype=np.int)
        sat_thk = plotutil.saturated_thickness(head, top, botm, laytyp,
                                               [hnoflo, hdry])

        # Calculate specific discharge
        qx, qy, qz = plotutil.centered_specific_discharge(frf, fff, flf, delr,
                                                          delc, sat_thk)

        # Select correct slice
        u = qx[self.layer, :, :]
        v = qy[self.layer, :, :]
        # apply step

        try:
            xcentergrid = self.sr.xcentergrid
            ycentergrid = self.sr.ycentergrid
        except AttributeError:
            xcentergrid = self.mg.xcell_centers()
            ycentergrid = self.mg.ycell_centers()

        x = xcentergrid[::istep, ::jstep]
        y = ycentergrid[::istep, ::jstep]
        u = u[::istep, ::jstep]
        v = v[::istep, ::jstep]
        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # mask discharge in inactive cells
        idx = (ib[self.layer, ::istep, ::jstep] == 0)
        u[idx] = np.nan
        v[idx] = np.nan

        # Rotate and plot
        urot, vrot = self.sr.rotate(u, v, self.sr.rotation)
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)

        return quiver

    def plot_pathline(self, pl, travel_time=None, **kwargs):
        """
        Plot the MODPATH pathlines.

        Parameters
        ----------
        pl : list of rec arrays or a single rec array
            rec array or list of rec arrays is data returned from
            modpathfile PathlineFile get_data() or get_alldata()
            methods. Data in rec array is 'x', 'y', 'z', 'time',
            'k', and 'particleid'.
        travel_time: float or str
            travel_time is a travel time selection for the displayed
            pathlines. If a float is passed then pathlines with times
            less than or equal to the passed time are plotted. If a
            string is passed a variety logical constraints can be added
            in front of a time value to select pathlines for a select
            period of time. Valid logical constraints are <=, <, >=, and
            >. For example, to select all pathlines less than 10000 days
            travel_time='< 10000' would be passed to plot_pathline.
            (default is None)
        kwargs : layer, ax, colors.  The remaining kwargs are passed
            into the LineCollection constructor. If layer='all',
            pathlines are output for all layers

        Returns
        -------
        lc : matplotlib.collections.LineCollection

        """
        err_msg = "plot_pathline() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def plot_endpoint(self, ep, direction='ending',
                      selection=None, selection_direction=None, **kwargs):
        """
        Plot the MODPATH endpoints.

        Parameters
        ----------
        ep : rec array
            A numpy recarray with the endpoint particle data from the
            MODPATH 6 endpoint file
        direction : str
            String defining if starting or ending particle locations should be
            considered. (default is 'ending')
        selection : tuple
            tuple that defines the zero-base layer, row, column location
            (l, r, c) to use to make a selection of particle endpoints.
            The selection could be a well location to determine capture zone
            for the well. If selection is None, all particle endpoints for
            the user-sepcified direction will be plotted. (default is None)
        selection_direction : str
            String defining is a selection should be made on starting or
            ending particle locations. If selection is not None and
            selection_direction is None, the selection direction will be set
            to the opposite of direction. (default is None)

        kwargs : ax, c, s or size, colorbar, colorbar_label, shrink. The
            remaining kwargs are passed into the matplotlib scatter
            method. If colorbar is True a colorbar will be added to the plot.
            If colorbar_label is passed in and colorbar is True then
            colorbar_label will be passed to the colorbar set_label()
            method. If shrink is passed in and colorbar is True then
            the colorbar size will be set using shrink.

        Returns
        -------
        sp : matplotlib.pyplot.scatter

        """
        direction = direction.lower()
        if direction == 'starting':
            xp, yp = 'x0', 'y0'

        elif direction == 'ending':
            xp, yp = 'x', 'y'

        else:
            errmsg = 'flopy.map.plot_endpoint direction must be "ending" ' + \
                     'or "starting".'
            raise Exception(errmsg)

        if selection_direction is not None:
            if selection_direction.lower() != 'starting' and \
                    selection_direction.lower() != 'ending':
                errmsg = 'flopy.map.plot_endpoint selection_direction ' + \
                         'must be "ending" or "starting".'
                raise Exception(errmsg)
        else:
            if direction.lower() == 'starting':
                selection_direction = 'ending'
            elif direction.lower() == 'ending':
                selection_direction = 'starting'

        if selection is not None:
            try:
                k, i, j = selection[0], selection[1], selection[2]
                if selection_direction.lower() == 'starting':
                    ksel, isel, jsel = 'k0', 'i0', 'j0'
                elif selection_direction.lower() == 'ending':
                    ksel, isel, jsel = 'k', 'i', 'j'
            except:
                errmsg = 'flopy.map.plot_endpoint selection must be a ' + \
                         'zero-based layer, row, column tuple (l, r, c) ' + \
                         'of the location to evaluate (i.e., well location).'
                raise Exception(errmsg)

        if selection is not None:
            idx = (ep[ksel] == k) & (ep[isel] == i) & (ep[jsel] == j)
            tep = ep[idx]
        else:
            tep = ep.copy()

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # scatter kwargs that users may redefine
        if 'c' not in kwargs:
            c = tep['finaltime'] - tep['initialtime']
        else:
            c = np.empty((tep.shape[0]), dtype="S30")
            c.fill(kwargs.pop('c'))

        s = 50
        if 's' in kwargs:
            s = float(kwargs.pop('s')) ** 2.
        elif 'size' in kwargs:
            s = float(kwargs.pop('size')) ** 2.

        # colorbar kwargs
        createcb = False
        if 'colorbar' in kwargs:
            createcb = kwargs.pop('colorbar')

        colorbar_label = 'Endpoint Time'
        if 'colorbar_label' in kwargs:
            colorbar_label = kwargs.pop('colorbar_label')

        shrink = 1.
        if 'shrink' in kwargs:
            shrink = float(kwargs.pop('shrink'))

        # rotate data
        x0r, y0r = self.sr.rotate(tep[xp], tep[yp], self.sr.rotation, 0.,
                                  self.mg.yedge[0])
        x0r += self.sr.xul
        y0r += self.sr.yul - self.mg.yedge[0]
        # build array to plot
        arr = np.vstack((x0r, y0r)).T

        # plot the end point data
        # todo: try ax.scatter to preseve the current axis object
        sp = ax.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)
        # sp = plt.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)

        # add a colorbar for travel times
        if createcb:
            cb = plt.colorbar(sp, shrink=shrink)
            cb.set_label(colorbar_label)
        return sp


class VertexModelMap(object):
    """
    Class to create a map of the model. Delegates plotting
    functionality based on model grid type.

    Parameters
    ----------
    sr : flopy.utils.reference.SpatialReference
        The spatial reference class (Default is None)
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy.modflow.ModflowDis object
        flopy discretization object. (Default is None)
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner.  The default is the sum of the
        delc array.
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    Notes
    -----
    ModelMap must know the position and rotation of the grid in order to make
    the plot.  This information is contained in the SpatialReference class
    (sr), which can be passed.  If sr is None, then it looks for sr in dis.
    If dis is None, then it looks for sr in model.dis.  If all of these
    arguments are none, then it uses xul, yul, and rotation.  If none of these
    arguments are provided, then it puts the lower-left-hand corner of the
    grid at (0, 0).

    """
    def __init__(self, sr=None, ax=None, model=None, dis=None, layer=0,
                 extent=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., length_multiplier=1.):

        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise Exception(s)

        self.model = model
        self.dis = dis
        self.layer = layer
        self.sr = None

        # todo: rewrite for modelgrid instance!
        if sr is not None:
            self.sr = copy.deepcopy(sr)

        elif dis is not None:
            self.sr = copy.deepcopy(dis.sr)

        elif model is not None:
            self.sr = copy.deepcopy(model.dis.sr)

        else:
            self.sr = SpatialReference(xll=xll, yll=yll, xul=xul, yul=yul,
                                       rotation=rotation,
                                       length_multiplier=length_multiplier)

        # model map override spatial reference settings
        if any(elem is not None for elem in (xul, yul, xll, yll)) or \
                rotation != 0 or length_multiplier != 1.:
            self.sr.length_multiplier = length_multiplier
            self.sr.set_spatialreference(xul, yul, xll, yll, rotation)

        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect('equal')
            except:
                self.ax = plt.subplot(1, 1, 1, aspect='equal', axisbg="white")
        else:
            self.ax = ax

        if extent is not None:
            self._extent = extent
        else:
            self._extent = None

    @property
    def extent(self):
        raise NotImplementedError()

    def plot_array(self, a, masked_values=None, **kwargs):
        """
        Plot an array.  If the array is two-dimensional, then the method
        will plot the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        patch_collection : matplotlib.collections.PatchCollection

        """
        nf = True
        if nf:
            raise NotImplementedError()

        if a.ndim == 2:
            plotarray = a[self.layer, :]
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1 or 2')

        mask = [False]*plotarray.size
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
                mask = np.ma.getmask(plotarray)

        if type(mask) is np.bool_:
            mask = [False] * plotarray.size

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # todo: solve this tabular data issue. We need a dictionary or dataframe of verticies to plot
        # todo: vertex and unstructured grids! Name the vertex table appropriately too!
        vertexdict = self.sr.xydict

        p = self.get_patch_collection(vertexdict, plotarray, mask, **kwargs)
        patch_collection = ax.add_collection(p)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return patch_collection

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is two-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        from scipy.interpolate import griddata

        if a.ndim == 2:
            plotarray = a[self.layer, :]
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1 or 2')

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' in kwargs.keys():
            if 'cmap' in kwargs.keys():
                cmap = kwargs.pop('cmap')
            cmap = None

        # todo: check that these data still exist within the sr class
        x = self.sr.xcenter_array
        y = self.sr.ycenter_array

        xi = np.linspace(np.min(x), np.max(x), 1000)
        yi = np.linspace(np.min(y), np.max(y), 1000)

        zi = griddata((x, y), plotarray, (xi[None, :], yi[:, None]), method='cubic')

        contour_set = ax.contour(xi, yi, zi, **kwargs)
        # contour_set = ax.contourf(xi, yi, zi, **kwargs)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self, ibound=None, color_noflow='black', **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound
        from the self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)

        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        raise NotImplementedError()

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        raise NotImplementedError()

    def plot_grid(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
        kwargs : ax, colors.  The remaining kwargs are passed into the
            the LineCollection constructor.

        Returns
        -------
        lc : matplotlib.collections.LineCollection

        """
        nf = True
        if nf:
            raise NotImplementedError()

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = '0.5'

        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'none'

        # todo: develop SR or modelgrid method to get the verticies in tabular
        # todo: or dictionary format. Can then use that for plotting.
        vertexdict = self.sr.xydict
        pc = self.get_patch_collection(vertexdict, grid=True, **kwargs)

        ax.add_collection(pc)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return pc

    def plot_bc(self, ftype=None, package=None, kper=0, color=None,
                plotAll=False, **kwargs):
        """
        Plot boundary conditions locations for a specific boundary
        type from a flopy model

        Parameters
        ----------
        ftype : string
            Package name string ('WEL', 'GHB', etc.). (Default is None)
        package : flopy.modflow.Modflow package class instance
            flopy package class instance. (Default is None)
        kper : int
            Stress period to plot
        color : string
            matplotlib color string. (Default is None)
        plotAll : bool
            Boolean used to specify that boundary condition locations for all
            layers will be plotted on the current ModelMap layer.
            (Default is False)
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # Find package to plot
        if package is not None:
            p = package
        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.model.get_package(ftype)
        else:
            raise Exception('Cannot find package to plot')

        # Get the list data
        try:
            # todo: remove test case from try statement, update to flopy code
            mflist = p.data[kper]
            # mflist = p.stress_period_data[kper]
        except Exception as e:
            raise Exception('Not a list-style boundary package:' + str(e))

        # Return if MfList is None
        if mflist is None:
            return None

        # todo: check that these are still valid (nlay, ncpl). Spatial reference has changed!
        nlay = self.sr.nlay
        # Plot the list locations
        plotarray = np.zeros((nlay, self.sr.ncpl), dtype=np.int)
        if plotAll:
            # todo: check if raw data is zero based or 1 based remove <-1> if appropriate
            idx = [mflist['ncpl'] - 1]
            # plotarray[:, idx] = 1
            pa = np.zeros((self.sr.ncpl), dtype=np.int)
            pa[idx] = 1
            for k in range(nlay):
                plotarray[k, :] = pa.copy()
        else:
            # todo: check if raw data is zero based or 1 based remove <-1> if appropriate
            idx = [mflist['layer'] - 1, mflist['ncpl'] - 1]

            plotarray[idx] = 1

        if color is None:
            if ftype in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[ftype]
            else:
                c = plotutil.bc_color_dict['default']
        else:
            c = color
        cmap = matplotlib.colors.ListedColormap(['0', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patch_collection = self.plot_array(plotarray, cmap=cmap, norm=norm, masked_values=[0], **kwargs)
        return patch_collection

    def plot_shapefile(self, shp, **kwargs):
        """
        Plot a shapefile.  The shapefile must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        shp : string
            Name of the shapefile to plot

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_shapefile()

        """
        err_msg = "plot_shapefile() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def plot_cvfd(self, verts, iverts, **kwargs):
        """
        Plot a cvfd grid.  The vertices must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        verts : ndarray
            2d array of x and y points.
        iverts : list of lists
            should be of len(ncells) with a list of vertex number for each cell

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_cvfd()

        """
        err_msg = "plot_cvfd() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def contour_array_cvfd(self, vertc, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        vertc : np.ndarray
            Array with of size (nc, 2) with centroid location of cvfd
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        err_msg = "contour_array_cvfd() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def plot_discharge(self, frf, fff, dis=None, flf=None, head=None, istep=1,
                       jstep=1, normalize=False, **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        frf : numpy.ndarray
            MODFLOW's 'flow right face'
        fff : numpy.ndarray
            MODFLOW's 'flow front face'
        flf : numpy.ndarray
            MODFLOW's 'flow lower face' (Default is None.)
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        istep : int
            row frequency to plot. (Default is 1.)
        jstep : int
            column frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors of specific discharge.

        """
        raise NotImplementedError()

    def plot_pathline(self, pl, travel_time=None, **kwargs):
        """
        Plot the MODPATH pathlines.

        Parameters
        ----------
        pl : list of rec arrays or a single rec array
            rec array or list of rec arrays is data returned from
            modpathfile PathlineFile get_data() or get_alldata()
            methods. Data in rec array is 'x', 'y', 'z', 'time',
            'k', and 'particleid'.
        travel_time: float or str
            travel_time is a travel time selection for the displayed
            pathlines. If a float is passed then pathlines with times
            less than or equal to the passed time are plotted. If a
            string is passed a variety logical constraints can be added
            in front of a time value to select pathlines for a select
            period of time. Valid logical constraints are <=, <, >=, and
            >. For example, to select all pathlines less than 10000 days
            travel_time='< 10000' would be passed to plot_pathline.
            (default is None)
        kwargs : layer, ax, colors.  The remaining kwargs are passed
            into the LineCollection constructor. If layer='all',
            pathlines are output for all layers

        Returns
        -------
        lc : matplotlib.collections.LineCollection

        """
        err_msg = "plot_pathline() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

    def plot_endpoint(self, ep, direction='ending',
                      selection=None, selection_direction=None, **kwargs):
        """
        Plot the MODPATH endpoints.

        Parameters
        ----------
        ep : rec array
            A numpy recarray with the endpoint particle data from the
            MODPATH 6 endpoint file
        direction : str
            String defining if starting or ending particle locations should be
            considered. (default is 'ending')
        selection : tuple
            tuple that defines the zero-base layer, row, column location
            (l, node) to use to make a selection of particle endpoints.
            The selection could be a well location to determine capture zone
            for the well. If selection is None, all particle endpoints for
            the user-sepcified direction will be plotted. (default is None)
        selection_direction : str
            String defining is a selection should be made on starting or
            ending particle locations. If selection is not None and
            selection_direction is None, the selection direction will be set
            to the opposite of direction. (default is None)

        kwargs : ax, c, s or size, colorbar, colorbar_label, shrink. The
            remaining kwargs are passed into the matplotlib scatter
            method. If colorbar is True a colorbar will be added to the plot.
            If colorbar_label is passed in and colorbar is True then
            colorbar_label will be passed to the colorbar set_label()
            method. If shrink is passed in and colorbar is True then
            the colorbar size will be set using shrink.

        Returns
        -------
        sp : matplotlib.pyplot.scatter

        """
        direction = direction.lower()
        if direction == 'starting':
            xp, yp = 'x0', 'y0'

        elif direction == 'ending':
            xp, yp = 'x', 'y'

        else:
            errmsg = 'flopy.map.plot_endpoint direction must be "ending" ' + \
                     'or "starting".'
            raise Exception(errmsg)

        if selection_direction is not None:
            if selection_direction.lower() != 'starting' and \
                    selection_direction.lower() != 'ending':
                errmsg = 'flopy.map.plot_endpoint selection_direction ' + \
                         'must be "ending" or "starting".'
                raise Exception(errmsg)
        else:
            if direction.lower() == 'starting':
                selection_direction = 'ending'
            elif direction.lower() == 'ending':
                selection_direction = 'starting'

        if selection is not None:
            try:
                # todo: check that this is the proper fmt!
                k, i = selection[0], selection[1]
                if selection_direction.lower() == 'starting':
                    ksel, node = 'k0', "node0"  # 'i0', 'j0'
                elif selection_direction.lower() == 'ending':
                    ksel, node = 'k', 'node'  # , 'j'
            except:
                errmsg = 'flopy.map.plot_endpoint selection must be a ' + \
                         'zero-based layer, node tuple (l, node) ' + \
                         'of the location to evaluate (i.e., well location).'
                raise Exception(errmsg)

        if selection is not None:
            # todo: check that this is the proper fmt!
            idx = (ep[ksel] == k) & (ep[node] == i)
            tep = ep[idx]
        else:
            tep = ep.copy()

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # scatter kwargs that users may redefine
        if 'c' not in kwargs:
            c = tep['finaltime'] - tep['initialtime']
        else:
            c = np.empty((tep.shape[0]), dtype="S30")
            c.fill(kwargs.pop('c'))

        s = 50
        if 's' in kwargs:
            s = float(kwargs.pop('s')) ** 2.
        elif 'size' in kwargs:
            s = float(kwargs.pop('size')) ** 2.

        # colorbar kwargs
        createcb = False
        if 'colorbar' in kwargs:
            createcb = kwargs.pop('colorbar')

        colorbar_label = 'Endpoint Time'
        if 'colorbar_label' in kwargs:
            colorbar_label = kwargs.pop('colorbar_label')

        shrink = 1.
        if 'shrink' in kwargs:
            shrink = float(kwargs.pop('shrink'))

        # rotate data
        x0r, y0r = self.sr.rotate(tep[xp], tep[yp], self.sr.rotation, 0.,
                                  self.sr.yedge[0])
        x0r += self.sr.xul
        y0r += self.sr.yul - self.sr.yedge[0]
        # build array to plot
        arr = np.vstack((x0r, y0r)).T

        # plot the end point data
        # todo: try ax.scatter to preseve the current axis object
        sp = ax.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)
        # sp = plt.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)

        # add a colorbar for travel times
        if createcb:
            cb = plt.colorbar(sp, shrink=shrink)
            cb.set_label(colorbar_label)
        return sp


class ModelMap(object):
    """
    Pending Depreciation: ModelMap acts as a PlotMapView factory
    object. Please migrate to PlotMapView for plotting
    functionality and future code compatibility

    Parameters
    ----------
    sr : flopy.utils.reference.SpatialReference
        The spatial reference class (Default is None)
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy.modflow.ModflowDis object
        flopy discretization object. (Default is None)
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner.  The default is the sum of the
        delc array.
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    Notes
    -----
    ModelMap must know the position and rotation of the grid in order to make
    the plot.  This information is contained in the SpatialReference class
    (sr), which can be passed.  If sr is None, then it looks for sr in dis.
    If dis is None, then it looks for sr in model.dis.  If all of these
    arguments are none, then it uses xul, yul, and rotation.  If none of these
    arguments are provided, then it puts the lower-left-hand corner of the
    grid at (0, 0).
    """
    def __new__(cls, sr=None, ax=None, model=None, dis=None, layer=0,
                extent=None, xul=None, yul=None, xll=None, yll=None,
                rotation=0., length_multiplier=1.):

        from ..plot import PlotMapView
        err_msg = "ModelMap will be replaced by " \
                  "PlotMapView(); Calling PlotMapView()"
        warnings.warn(err_msg, PendingDeprecationWarning)

        return PlotMapView(sr=sr, ax=ax, model=model, dis=dis, layer=layer,
                           extent=extent, xul=xul, yul=yul, xll=xll, yll=yll,
                           rotation=rotation, length_multiplier=length_multiplier)


