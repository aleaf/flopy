import os
import numpy as np
import flopy
from flopy.utils.lgrutil import Lgr


tpth = os.path.join('temp', 't063')
# make the directory if it does not exist
if not os.path.isdir(tpth):
    os.makedirs(tpth)


def test_lgrutil():
    nlayp = 5
    nrowp = 5
    ncolp = 5
    delrp = 100.
    delcp = 100.
    topp = 100.
    botmp = [-100, -200, -300, -400, -500]
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[0:2, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = [1, 1, 0, 0, 0]

    lgr = Lgr(nlayp, nrowp, ncolp, delrp, delcp, topp, botmp,
              idomainp, ncpp=ncpp, ncppl=ncppl, xllp=100., yllp=100.)

    # child shape
    assert lgr.get_shape() == (2, 9, 9), 'child shape is not (2, 9, 9)'

    # child delr/delc
    delr, delc = lgr.get_delr_delc()
    assert np.allclose(delr, delrp / ncpp), 'child delr not correct'
    assert np.allclose(delc, delcp / ncpp), 'child delc not correct'

    # child idomain
    idomain = lgr.get_idomain()
    assert idomain.min() == idomain.max() == 1
    assert idomain.shape == (2, 9, 9)

    # replicated parent array
    ap = np.arange(nrowp * ncolp).reshape((nrowp, ncolp))
    ac = lgr.get_replicated_parent_array(ap)
    assert ac[0, 0] == 6
    assert ac[-1, -1] == 18

    # child top/bottom
    topc, botmc = lgr.get_top_botm()
    assert topc.shape == (9, 9)
    assert botmc.shape == (2, 9, 9)
    assert topc.min() == topc.max() == 100.
    errmsg = '{} /= {}'.format(botmc[:, 0, 0], np.array(botmp[:2]))
    assert np.allclose(botmc[:, 0, 0], np.array(botmp[:2])), errmsg

    # exchange data
    exchange_data = lgr.get_exchange_data(angldegx=True, cdist=True)


    ans1 = [(0, 1, 0), (0, 0, 0), 1, 50.0, 16.666666666666668,
            33.333333333333336, 0.0, 354.33819375782156]
    errmsg = '{} /= {}'.format(ans1, exchange_data[0])
    assert exchange_data[0] == ans1, errmsg

    ans2 = [(2, 3, 3), (1, 8, 8), 0, 50.0, 50,
            1111.1111111111113, 180., 100.]
    errmsg = '{} /= {}'.format(ans2, exchange_data[-1])
    assert exchange_data[-1] == ans2, errmsg

    errmsg = 'exchanges should be 71 horizontal plus 81 vertical'
    assert len(exchange_data) == 72 + 81, errmsg

    # list of parent cells connected to a child cell
    assert lgr.get_parent_connections(0, 0, 0) == [((0, 1, 0), -1),
                                                   ((0, 0, 1), 2)]
    assert lgr.get_parent_connections(1, 8, 8) == [((1, 3, 4), 1),
                                                   ((1, 4, 3), -2),
                                                   ((2, 3, 3), -3)]

    return


def test_lgrutil_gnc():
    nlayp = 5
    nrowp = 5
    ncolp = 5
    delrp = 100.
    delcp = 100.
    topp = 100.
    botmp = [-100, -200, -300, -400, -500]
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[0:2, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = [1, 1, 0, 0, 0]

    lgr = Lgr(nlayp, nrowp, ncolp, delrp, delcp, topp, botmp,
              idomainp, ncpp=ncpp, ncppl=ncppl, xllp=100., yllp=100.)
    parentn, parentj, alphaj = zip(*lgr.get_gnc_parent_j_info(0, 0, 0))
    assert parentn == ((0, 1, 0), (0, 0, 1))
    assert parentj == ((0, 0, 0), (0, 0, 0))
    assert len(alphaj) == 2
    assert np.allclose(alphaj, 0.33333)
    
    # cell connection that coincides with i position of parent node
    assert lgr.get_gnc_parent_j_info(0, 1, 0) is None
    
    parentn, parentj, alphaj = zip(*lgr.get_gnc_parent_j_info(1, 8, 8))
    assert parentn == ((1, 3, 4), (1, 4, 3))
    assert parentj == ((1, 4, 4), (1, 4, 4))
    assert len(alphaj) == 2
    assert np.allclose(alphaj, 0.33333)
    
    gnclist = lgr.get_gnc_data()
    
    # create set of parent connections to check
    pconn_info = lgr.get_parent_connections(0, 0, 0)
    pconn = {pconn_info[0][0], pconn_info[1][0]}
    pconn_info = lgr.get_parent_connections(1, 8, 8)
    pconn.update({pconn_info[0][0], pconn_info[1][0]})
    
    for item in gnclist:
        cellidn, cellidm, parentj, alphaj = item
        #pconn_info = lgr.get_parent_connections(*cellidm)
        if cellidm == (0, 0, 0):
            pass
            pconn.remove(cellidn)
            assert parentj == (0, 0, 0)
            assert np.allclose(alphaj, 0.333333)
        elif cellidm == (1, 8, 8):
            pconn.remove(cellidn)
            assert parentj == (1, 4, 4)
            assert np.allclose(alphaj, 0.333333)
            
    # all parent connections should have been listed
    assert not any(pconn)
    
    assert False, "still need to implement bottom connections"

    

if __name__ == '__main__':
    #test_lgrutil()
    test_lgrutil_gnc()

