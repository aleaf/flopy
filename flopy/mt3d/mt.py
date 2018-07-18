import os
import sys
import numpy as np
from ..mbase import BaseModel
from ..pakbase import Package
from ..utils import mfreadnam
from .mtbtn import Mt3dBtn
from .mtadv import Mt3dAdv
from .mtdsp import Mt3dDsp
from .mtssm import Mt3dSsm
from .mtrct import Mt3dRct
from .mtgcg import Mt3dGcg
from .mttob import Mt3dTob
from .mtphc import Mt3dPhc
from .mtuzt import Mt3dUzt
from .mtsft import Mt3dSft
from .mtlkt import Mt3dLkt
from ..grid.reference import TemporalReference
from ..grid.modelgrid import SimulationTime
from ..grid.structuredmodelgrid import StructuredModelGrid

class Mt3dList(Package):
    """
    List package class
    """

    def __init__(self, model, extension='list', listunit=7):
        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, 'LIST', listunit)
        # self.parent.add_package(self) This package is not added to the base
        # model so that it is not included in get_name_file_entries()
        return

    def __repr__(self):
        return 'List package class'

    def write_file(self):
        # Not implemented for list class
        return


'''
class Mt3dms(BaseModel):
    'MT3DMS base class'

    def __init__(self, modelname='mt3dmstest', namefile_ext='nam',
                 modflowmodel=None, ftlfilename=None,
                 model_ws=None, external_path=None, verbose=False,
                 load=True, listunit=7, exe_name='mt3dms.exe', ):
        BaseModel.__init__(self, modelname, namefile_ext, model_ws=model_ws,
                           exe_name=exe_name)
        self.heading = '# Name file for MT3DMS, generated by Flopy.'
        self.__mf = modflowmodel
        self.lst = Mt3dList(self, listunit=listunit)
        self.ftlfilename = ftlfilename
        self.__adv = None
        self.__btn = None
        self.__dsp = None
        self.__gcg = None
        self.__rct = None
        self.__ssm = None
        self.array_free_format = False
        self.external_path = external_path
        self.external = False
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.load = load
        self.__next_ext_unit = 500
        if external_path is not None:
            if os.path.exists(external_path):
                print("Note: external_path " + str(external_path) + \
                      " already exists")
            # assert os.path.exists(external_path),'external_path does not exist'
            else:
                os.mkdir(external_path)
            self.external = True
        self.verbose = verbose
        return

    def __repr__(self):
        return 'MT3DMS model'

    def get_ncomp(self):
        btn = self.get_package('BTN')
        if (btn):
            return btn.ncomp
        else:
            return 1

    # function to encapsulate next_ext_unit attribute
    def next_ext_unit(self):
        self.__next_ext_unit += 1
        return self.__next_ext_unit

    def getadv(self):
        if (self.__adv == None):
            for p in (self.packagelist):
                if isinstance(p, Mt3dAdv):
                    self.__adv = p
        return self.__adv

    def getbtn(self):
        if (self.__btn == None):
            for p in (self.packagelist):
                if isinstance(p, Mt3dBtn):
                    self.__btn = p
        return self.__btn

    def getdsp(self):
        if (self.__dsp == None):
            for p in (self.packagelist):
                if isinstance(p, Mt3dDsp):
                    self.__dsp = p
        return self.__dsp

    def getgcg(self):
        if (self.__gcg == None):
            for p in (self.packagelist):
                if isinstance(p, Mt3dGcg):
                    self.__gcg = p
        return self.__gcg

    def getmf(self):
        return self.__mf

    def getrct(self):
        if (self.__rct == None):
            for p in (self.packagelist):
                if isinstance(p, Mt3dRct):
                    self.__rct = p
        return self.__rct

    def getssm(self):
        if (self.__ssm == None):
            for p in (self.packagelist):
                if isinstance(p, Mt3dSsm):
                    self.__ssm = p
        return self.__ssm

    def write_name_file(self):
        fn_path = os.path.join(self.model_ws, self.namefile)
        f_nam = open(fn_path, 'w')
        f_nam.write('%s\n' % (self.heading))
        f_nam.write('%s %3i %s\n' % (self.lst.name[0], self.lst.unit_number[0],
                                     self.lst.file_name[0]))
        if self.ftlfilename is not None:
            f_nam.write('%s %3i %s\n' % ('FTL', 39, self.ftlfilename))
        f_nam.write('%s' % self.get_name_file_entries())
        for u, f in zip(self.external_units, self.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u) + f + '\n')
        f_nam.close()

    adv = property(getadv)  # Property has no setter, so read-only
    btn = property(getbtn)  # Property has no setter, so read-only
    dsp = property(getdsp)  # Property has no setter, so read-only
    gcg = property(getgcg)  # Property has no setter, so read-only
    mf = property(getmf)  # Property has no setter, so read-only
    rct = property(getrct)  # Property has no setter, so read-only
    ssm = property(getssm)  # Property has no setter, so read-only
    ncomp = property(get_ncomp)
'''


class Mt3dms(BaseModel):
    """
    MT3DMS Model Class.

    Parameters
    ----------
    modelname : string, optional
        Name of model.  This string will be used to name the MODFLOW input
        that are created with write_model. (the default is 'mt3dtest')
    namefile_ext : string, optional
        Extension for the namefile (the default is 'nam')
    modflowmodel : flopy.modflow.mf.Modflow
        This is a flopy Modflow model object upon which this Mt3dms model
        is based. (the default is None)
    version : string, optional
        Version of MT3DMS to use (the default is 'mt3dms').
    exe_name : string, optional
        The name of the executable to use (the default is
        'mt3dms.exe').
    listunit : integer, optional
        Unit number for the list file (the default is 2).
    model_ws : string, optional
        model workspace.  Directory name to create model data sets.
        (default is the present working directory).
    external_path : string
        Location for external files (default is None).
    verbose : boolean, optional
        Print additional information to the screen (default is False).
    load : boolean, optional
         (default is True).
    silent : integer
        (default is 0)

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.mt3d.mt.Mt3dms()

    """

    def __init__(self, modelname='mt3dtest', namefile_ext='nam',
                 modflowmodel=None, ftlfilename="mt3d_link.ftl", ftlfree=False,
                 version='mt3dms', exe_name='mt3dms.exe',
                 structured=True, listunit=None, ftlunit=None,
                 model_ws='.', external_path=None,
                 verbose=False, load=True, silent=0):

        # Call constructor for parent object
        BaseModel.__init__(self, modelname, namefile_ext, exe_name, model_ws,
                           structured=structured)

        # Set attributes
        self.version_types = {'mt3dms': 'MT3DMS', 'mt3d-usgs': 'MT3D-USGS'}

        self.set_version(version.lower())

        if listunit is None:
            listunit = 16

        if ftlunit is None:
            ftlunit = 10

        self.lst = Mt3dList(self, listunit=listunit)
        self.mf = modflowmodel
        self.ftlfilename = ftlfilename
        self.ftlfree = ftlfree
        self.ftlunit = ftlunit

        # Check whether specified ftlfile exists in model directory; if not,
        # warn user
        if os.path.isfile(os.path.join(self.model_ws,
                                       str(modelname + '.' + namefile_ext))):
            with open(os.path.join(self.model_ws, str(
                                    modelname + '.' + namefile_ext))) as nm_file:
                for line in nm_file:
                    if line[0:3] == 'FTL':
                        ftlfilename = line.strip().split()[2]
                        break
        if ftlfilename is None:
            print("User specified FTL file does not exist in model directory")
            print("MT3D will not work without a linker file")
        else:
            if os.path.isfile(os.path.join(self.model_ws, ftlfilename)):
                # Check that the FTL present in the directory is of the format
                # specified by the user, i.e., is same as ftlfree
                # Do this by checking whether the first non-blank character is
                # an apostrophe.
                # If code lands here, then ftlfilename exists, open and read
                # first 4 characters
                f = open(os.path.join(self.model_ws, ftlfilename), 'rb')
                c = f.read(4)
                if isinstance(c, bytes):
                    c = c.decode()

                # if first non-blank char is an apostrophe, then formatted,
                # otherwise binary
                if (c.strip()[0] == "'" and self.ftlfree) or \
                        (c.strip()[0] != "'" and not self.ftlfree):
                    pass
                else:
                    msg = "Specified value of ftlfree conflicts with FTL " + \
                          "file format"
                    print(msg)
                    msg = 'Switching ftlfree from ' + \
                          '{} '.format(str(self.ftlfree)) + \
                          'to {}'.format(str(not self.ftlfree))
                    print(msg)
                    self.ftlfree = not self.ftlfree  # Flip the bool

        # external option stuff
        self.array_free_format = False
        self.array_format = 'mt3d'
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.external = False
        self.verbose = verbose
        self.load = load
        # the starting external data unit number
        self._next_ext_unit = 2000
        if external_path is not None:
            #assert model_ws == '.', "ERROR: external cannot be used " + \
            #                        "with model_ws"

            # external_path = os.path.join(model_ws, external_path)
            if os.path.exists(external_path):
                print("Note: external_path " + str(external_path) +
                      " already exists")
            # assert os.path.exists(external_path),'external_path does not exist'
            else:
                os.mkdir(external_path)
            self.external = True
        self.external_path = external_path
        self.verbose = verbose
        self.silent = silent

        # Create a dictionary to map package with package object.
        # This is used for loading models.
        self.mfnam_packages = {
            'btn': Mt3dBtn,
            'adv': Mt3dAdv,
            'dsp': Mt3dDsp,
            'ssm': Mt3dSsm,
            'rct': Mt3dRct,
            'gcg': Mt3dGcg,
            'tob': Mt3dTob,
            'phc': Mt3dPhc,
            'lkt': Mt3dLkt,
            'sft': Mt3dSft,
            'uzt': Mt3dUzt
        }
        return

    def __repr__(self):
        return 'MT3DMS model'

    @property
    def modelgrid(self):
        tr = TemporalReference(self.mf.dis.itmuni, self.mf.start_datetime)
        data_frame = {'perlen': self.mf.dis.perlen.array,
                      'nstp': self.mf.dis.nstp.array,
                      'tsmult': self.mf.dis.tsmult.array}
        sim_time = SimulationTime(data_frame,
                                  self.mf.dis.itmuni_dict[
                                  self.mf.dis.itmuni], tr)
        if self.mf.bas is not None:
            ibound = self.btn.icbund.array
        else:
            ibound = None
        return StructuredModelGrid(self.mf.dis.delc.array,
                                   self.mf.dis.delr.array,
                                   self.mf.dis.top.array,
                                   self.mf.dis.botm.array, ibound,
                                   self.mf.sr, sim_time,
                                   self.mf.name,
                                   self.mf.dis.steady.array)

    @property
    def solver_tols(self):
        if self.gcg is not None:
            return self.gcg.cclose, -999
        return None

    @property
    def sr(self):
        if self.mf is not None:
            return self.mf.sr
        return None

    @property
    def nlay(self):
        if (self.btn):
            return self.btn.nlay
        else:
            return 0

    @property
    def nrow(self):
        if (self.btn):
            return self.btn.nrow
        else:
            return 0

    @property
    def ncol(self):
        if (self.btn):
            return self.btn.ncol
        else:
            return 0

    @property
    def nper(self):
        if (self.btn):
            return self.btn.nper
        else:
            return 0

    @property
    def ncomp(self):
        if (self.btn):
            return self.btn.ncomp
        else:
            return 1

    @property
    def mcomp(self):
        if (self.btn):
            return self.btn.mcomp
        else:
            return 1

    def get_nrow_ncol_nlay_nper(self):
        if (self.btn):
            return self.btn.nrow, self.btn.ncol, self.btn.nlay, self.btn.nper
        else:
            return 0, 0, 0, 0

    # Property has no setter, so read-only
    nrow_ncol_nlay_nper = property(get_nrow_ncol_nlay_nper)

    def write_name_file(self):
        """
        Write the name file.

        """
        fn_path = os.path.join(self.model_ws, self.namefile)
        f_nam = open(fn_path, 'w')
        f_nam.write('{}\n'.format(self.heading))
        f_nam.write('{:14s} {:5d}  {}\n'.format(self.lst.name[0],
                                                self.lst.unit_number[0],
                                                self.lst.file_name[0]))
        if self.ftlfilename is not None:
            ftlfmt = ''
            if self.ftlfree:
                ftlfmt = 'FREE'
            f_nam.write('{:14s} {:5d}  {} {}\n'.format('FTL', self.ftlunit,
                                                       self.ftlfilename,
                                                       ftlfmt))
        # write file entries in name file
        f_nam.write('{}'.format(self.get_name_file_entries()))

        # write the external files
        for u, f in zip(self.external_units, self.external_fnames):
            f_nam.write('DATA           {0:5d}  '.format(u) + f + '\n')

        # write the output files
        for u, f, b in zip(self.output_units, self.output_fnames,
                           self.output_binflag):
            if u == 0:
                continue
            if b:
                f_nam.write(
                    'DATA(BINARY)   {0:5d}  '.format(u) + f + ' REPLACE\n')
            else:
                f_nam.write('DATA           {0:5d}  '.format(u) + f + '\n')

        f_nam.close()
        return

    def load_results(self, **kwargs):
        return

    @staticmethod
    def load(f, version='mt3dms', exe_name='mt3dms.exe', verbose=False,
             model_ws='.', load_only=None, forgive=False, modflowmodel=None):
        """
        Load an existing model.

        Parameters
        ----------
        f : string
            Full path and name of MT3D name file.

        version : string
            The version of MT3D (mt3dms, or mt3d-usgs)
            (default is mt3dms)

        exe_name : string
            The name of the executable to use if this loaded model is run.
            (default is mt3dms.exe)

        verbose : bool
            Write information on the load process if True.
            (default is False)

        model_ws : string
            The path for the model workspace.
            (default is the current working directory '.')

        load_only : list of strings
            Filetype(s) to load (e.g. ['btn', 'adv'])
            (default is None, which means that all will be loaded)

        modflowmodel : flopy.modflow.mf.Modflow
            This is a flopy Modflow model object upon which this Mt3dms
            model is based. (the default is None)

        Returns
        -------
        mt : flopy.mt3d.mt.Mt3dms
            flopy Mt3d model object

        Notes
        -----
        The load method does not retain the name for the MODFLOW-generated
        FTL file.  This can be added manually after the MT3D model has been
        loaded.  The syntax for doing this manually is
        mt.ftlfilename = 'example.ftl'

        Examples
        --------

        >>> import flopy
        >>> f = 'example.nam'
        >>> mt = flopy.mt3d.mt.Mt3dms.load(f)
        >>> mt.ftlfilename = 'example.ftl'

        """
        # test if name file is passed with extension (i.e., is a valid file)
        modelname_extension = None
        if os.path.isfile(os.path.join(model_ws, f)):
            modelname = f.rpartition('.')[0]
            modelname_extension = f.rpartition('.')[2]
        else:
            modelname = f

        if verbose:
            sys.stdout.write('\nCreating new model with name: {}\n{}\n\n'.
                             format(modelname, 50 * '-'))
        mt = Mt3dms(modelname=modelname, namefile_ext=modelname_extension,
                    version=version, exe_name=exe_name,
                    verbose=verbose, model_ws=model_ws,
                    modflowmodel=modflowmodel)

        files_succesfully_loaded = []
        files_not_loaded = []

        # read name file
        try:
            # namefile_path = os.path.join(mt.model_ws, mt.namefile)
            # namefile_path = f
            namefile_path = os.path.join(mt.model_ws, f)
            ext_unit_dict = mfreadnam.parsenamefile(namefile_path,
                                                    mt.mfnam_packages,
                                                    verbose=verbose)
        except Exception as e:
            # print("error loading name file entries from file")
            # print(str(e))
            # return None
            raise Exception(
                "error loading name file entries from file:\n" + str(e))

        if mt.verbose:
            print('\n{}\nExternal unit dictionary:\n{}\n{}\n'.
                  format(50 * '-', ext_unit_dict, 50 * '-'))

        # reset unit number for list file
        unitnumber = None
        for key, value in ext_unit_dict.items():
            if value.filetype == 'LIST':
                unitnumber = key
                filepth = os.path.basename(value.filename)
        if unitnumber == 'LIST':
            unitnumber = 16
        if unitnumber is not None:
            mt.lst.unit_number = [unitnumber]
            mt.lst.file_name = [filepth]

        # set ftl information
        unitnumber = None
        for key, value in ext_unit_dict.items():
            if value.filetype == 'FTL':
                unitnumber = key
                filepth = os.path.basename(value.filename)
        if unitnumber == 'FTL':
            unitnumber = 10
        if unitnumber is not None:
            mt.ftlunit = unitnumber
            mt.ftlfilename = filepth

        # load btn
        btn = None
        btn_key = None
        for key, item in ext_unit_dict.items():
            if item.filetype.lower() == "btn":
                btn = item
                btn_key = key
                break

        if btn is None:
            return None

        try:
            pck = btn.package.load(btn.filename, mt,
                                   ext_unit_dict=ext_unit_dict)
        except Exception as e:
            raise Exception('error loading BTN: {0}'.format(str(e)))
        files_succesfully_loaded.append(btn.filename)
        if mt.verbose:
            sys.stdout.write('   {:4s} package load...success\n'
                             .format(pck.name[0]))
        ext_unit_dict.pop(btn_key)

        if load_only is None:
            load_only = []
            for key, item in ext_unit_dict.items():
                load_only.append(item.filetype)
        else:
            if not isinstance(load_only, list):
                load_only = [load_only]
            not_found = []
            for i, filetype in enumerate(load_only):
                filetype = filetype.upper()
                if filetype != 'BTN':
                    load_only[i] = filetype
                    found = False
                    for key, item in ext_unit_dict.items():
                        if item.filetype == filetype:
                            found = True
                            break
                    if not found:
                        not_found.append(filetype)
            if len(not_found) > 0:
                raise Exception(
                    "the following load_only entries were not found "
                    "in the ext_unit_dict: " + ','.join(not_found))

        # try loading packages in ext_unit_dict
        for key, item in ext_unit_dict.items():
            if item.package is not None:
                if item.filetype in load_only:
                    if forgive:
                        try:
                            pck = item.package.load(item.filename, mt,
                                                    ext_unit_dict=ext_unit_dict)
                            files_succesfully_loaded.append(item.filename)
                            if mt.verbose:
                                sys.stdout.write(
                                    '   {:4s} package load...success\n'
                                        .format(pck.name[0]))
                        except BaseException as o:
                            if mt.verbose:
                                sys.stdout.write(
                                    '   {:4s} package load...failed\n   {!s}\n'
                                        .format(item.filetype, o))
                            files_not_loaded.append(item.filename)
                    else:
                        pck = item.package.load(item.filename, mt,
                                                ext_unit_dict=ext_unit_dict)
                        files_succesfully_loaded.append(item.filename)
                        if mt.verbose:
                            sys.stdout.write(
                                '   {:4s} package load...success\n'
                                    .format(pck.name[0]))
                else:
                    if mt.verbose:
                        sys.stdout.write('   {:4s} package load...skipped\n'
                                         .format(item.filetype))
                    files_not_loaded.append(item.filename)
            elif "data" not in item.filetype.lower():
                files_not_loaded.append(item.filename)
                if mt.verbose:
                    sys.stdout.write('   {:4s} package load...skipped\n'
                                     .format(item.filetype))
            elif "data" in item.filetype.lower():
                if mt.verbose:
                    sys.stdout.write('   {} file load...skipped\n      {}\n'
                                     .format(item.filetype,
                                             os.path.basename(item.filename)))
                if key not in mt.pop_key_list:
                    mt.external_fnames.append(item.filename)
                    mt.external_units.append(key)
                    mt.external_binflag.append("binary"
                                               in item.filetype.lower())
                    mt.external_output.append(False)

        # pop binary output keys and any external file units that are now
        # internal
        for key in mt.pop_key_list:
            try:
                mt.remove_external(unit=key)
                ext_unit_dict.pop(key)
            except:
                if mt.verbose:
                    sys.stdout.write('Warning: external file unit " +\
                        "{} does not exist in ext_unit_dict.\n'.format(key))

        # write message indicating packages that were successfully loaded
        if mt.verbose:
            print(1 * '\n')
            s = '   The following {0} packages were successfully loaded.' \
                .format(len(files_succesfully_loaded))
            print(s)
            for fname in files_succesfully_loaded:
                print('      ' + os.path.basename(fname))
            if len(files_not_loaded) > 0:
                s = '   The following {0} packages were not loaded.'.format(
                    len(files_not_loaded))
                print(s)
                for fname in files_not_loaded:
                    print('      ' + os.path.basename(fname))
                print('\n')

        # return model object
        return mt

    @staticmethod
    def load_mas(fname):
        """
        Load an mt3d mas file and return a numpy recarray

        Parameters
        ----------
        fname : str
            name of MT3D mas file

        Returns
        -------
        r : np.ndarray

        """
        if not os.path.isfile(fname):
            raise Exception('Could not find file: {}'.format(fname))
        dtype = [('time', float), ('total_in', float),
                 ('total_out', float),
                 ('sources', float), ('sinks', float),
                 ('fluid_storage', float),
                 ('total_mass', float), ('error_in-out', float),
                 ('error_alt', float)]
        r = np.loadtxt(fname, skiprows=2, dtype=dtype)
        r = r.view(np.recarray)
        return r

    @staticmethod
    def load_obs(fname):
        """
        Load an mt3d obs file and return a numpy recarray

        Parameters
        ----------
        fname : str
            name of MT3D obs file

        Returns
        -------
        r : np.ndarray

        """
        firstline = 'STEP   TOTAL TIME             LOCATION OF OBSERVATION POINTS (K,I,J)'
        dtype = [('step', int), ('time', float)]
        nobs = 0
        obs = []

        if not os.path.isfile(fname):
            raise Exception('Could not find file: {}'.format(fname))
        with open(fname, 'r') as f:
            line = f.readline()
            if line.strip() != firstline:
                msg = 'First line in file must be \n{}\nFound {}'.format(
                    firstline, line.strip())
                msg += '\n{} does not appear to be a valid MT3D OBS file'.format(
                    fname)
                raise Exception(msg)

            # Read obs names (when break, line will have first data line)
            nlineperrec = 0
            while True:
                line = f.readline()
                if line[0:7].strip() == '1':
                    break
                nlineperrec += 1
                ll = line.strip().split()
                while len(ll) > 0:
                    k = int(ll.pop(0))
                    i = int(ll.pop(0))
                    j = int(ll.pop(0))
                    obsnam = '({}, {}, {})'.format(k, i, j)
                    if obsnam in obs:
                        obsnam += str(len(obs) + 1)  # make obs name unique
                    obs.append(obsnam)

            icount = 0
            r = []
            while True:
                ll = []
                for n in range(nlineperrec):
                    icount += 1
                    if icount > 1:
                        line = f.readline()
                    ll.extend(line.strip().split())

                if not line:
                    break

                rec = [int(ll[0])]
                for val in ll[1:]:
                    rec.append(float(val))
                r.append(tuple(rec))

        # add obs names to dtype
        for nameob in obs:
            dtype.append((nameob, float))
        r = np.array(r, dtype=dtype)
        r = r.view(np.recarray)
        return r
