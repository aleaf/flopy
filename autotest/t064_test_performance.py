"""
Tests to prevent performance regressions
"""
import os
import shutil
import time
import numpy as np
import flopy.modflow as fm


class TestModflowPerformance():
    """Test flopy.modflow performance with realistic model/package sizes,
    in a reasonable timeframe.
    """
    @classmethod
    def setup_class(cls):
        """Make a modflow model."""
        print('setting up model...')
        t0 = time.time()
        size = 1000
        nlay = 2
        nper = 1
        nsfr = int((size ** 2)/1000)
        nwells = int(5e5)

        cls.modelname = 'junk'
        cls.model_ws = 'temp/t064'
        external_path = 'external/'

        if not os.path.isdir(cls.model_ws):
            os.makedirs(cls.model_ws)
        if not os.path.isdir(os.path.join(cls.model_ws, external_path)):
            os.makedirs(os.path.join(cls.model_ws, external_path))

        m = fm.Modflow(cls.modelname, model_ws=cls.model_ws, external_path=external_path)

        dis = fm.ModflowDis(m, nper=nper, nlay=nlay, nrow=size, ncol=size,
                            top=nlay, botm=list(range(nlay)))

        rch = fm.ModflowRch(m, rech={k: .001 - np.cos(k) * .001 for k in range(nper)})

        ra = fm.ModflowWel.get_empty(nwells)
        well_spd = {}
        for kper in range(nper):
            ra_per = ra.copy()
            ra_per['k'] = 1
            i = (np.ones((size, size)) * np.arange(size)).transpose().ravel().astype(int)
            j = list(range(size)) * size
            ra_per['i'] = i[:len(ra_per)]
            ra_per['j'] = j[:len(ra_per)]
            well_spd[kper] = ra_per
        wel = fm.ModflowWel(m, stress_period_data=well_spd)

        # SFR package
        rd = fm.ModflowSfr2.get_empty_reach_data(nsfr)
        rd['iseg'] = range(len(rd))
        rd['ireach'] = 1
        sd = fm.ModflowSfr2.get_empty_segment_data(nsfr)
        sd['nseg'] = range(len(sd))
        sfr = fm.ModflowSfr2(reach_data=rd, segment_data=sd, model=m)
        cls.init_time = time.time() - t0
        cls.m = m

    def test_init_time(self):
        """test model and package init time(s)."""
        mfp = TestModflowPerformance()
        target = 0.3 # seconds
        assert mfp.init_time < target, "model init took {:.2f}s, should take < {:.1f}s".format(mfp.init_time, target)
        print('setting up model took {:.2f}s'.format(mfp.init_time))

    def test_0_write_time(self):
        """test write time"""
        print('writing files...')
        mfp = TestModflowPerformance()
        target = 20
        t0 = time.time()
        mfp.m.write_input()
        t1 = time.time() - t0
        assert t1 < target, "model write took {:.2f}s, should take < {:.1f}s".format(t1, target)
        print('writing input took {:.2f}s'.format(t1))

    def test_9_load_time(self):
        """test model load time"""
        print('loading model...')
        mfp = TestModflowPerformance()
        target = 2
        t0 = time.time()
        m = fm.Modflow.load('{}.nam'.format(mfp.modelname),
                            model_ws=mfp.model_ws, check=False)
        t1 = time.time() - t0
        assert t1 < target, "model load took {:.2f}s, should take < {:.1f}s".format(t1, target)
        print('loading the model took {:.2f}s'.format(t1))

    def test_10_util2d_load(self):
        import io
        from flopy.utils import Util2d
        target = 0.5
        mfp = TestModflowPerformance()
        size = mfp.m.nrow

        f = io.StringIO("OPEN/CLOSE external/model_top.ref 1 (1000E15.6) -1 \n")
        t0 = time.time()
        result = Util2d.load(f, mfp.m, shape=(size, size), dtype=np.float32, name='junk')
        assert result.array.shape == (size, size)
        t1 = time.time() - t0
        assert t1 < target, "array load took {:.2f}s, should take < {:.1f}s".format(t1, target)
        print('loading a {0} x {0} array took {1:.2f}s'.format(size, t1))

    @classmethod
    def teardown_class(cls):
        # cleanup
        shutil.rmtree(cls.model_ws)
