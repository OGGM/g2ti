# Python imports
import unittest
import numpy as np
import os
import shutil
import xarray as xr
import oggm.cfg as cfg
from oggm.tests.funcs import get_test_dir
from oggm.utils import get_demo_file
from oggm.core import gis

import g2ti


class TestG2TI(unittest.TestCase):

    # Test case for a tidewater glacier

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_g2ti')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        self.idir = '/home/mowglie/disk/G2TI/data/geometry/RGI60-11/RGI60-11.00887'

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        gdf = cfg.PARAMS['intersects_gdf']
        gdf['RGIId_1'] = gdf['RGIId_1'].str.replace('RGI50', 'RGI60')
        gdf['RGIId_2'] = gdf['RGIId_2'].str.replace('RGI50', 'RGI60')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_domain(self):


        gdir = g2ti.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        g2ti.g2ti_masks(gdir)

    def test_g2ti_data(self):

        gdir = g2ti.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        g2ti.g2ti_masks(gdir)

        i, j, _, _, thick = g2ti.get_ref_gtd_data(gdir)

        ds = xr.open_dataset(gdir.get_filepath('gridded_data'))

        is_mask = ds.glacier_mask.isel_points(x=i, y=j)
        assert np.all(is_mask)

    def test_optim(self):

        gdir = g2ti.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        g2ti.g2ti_masks(gdir)
        out, ds = g2ti.optimize_distribute_thickness_single_glacier(gdir)

        np.testing.assert_allclose(out.bias.abs().min(), 0, atol=0.05)

        out_s = out.loc[out['bias'].abs() < 5]
        out_s = out_s.loc[out_s.idxmin()['mad']]

        bias = ds['oggm'] - ds['ref_thick']
        mbias = np.abs(bias.mean(dim='points'))
        mbias = mbias.min(dim=['fac_dis', 'fac_slope'])
        assert mbias.min() < 1

    def test_distribute(self):

        gdir = g2ti.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        gis.glacier_masks(gdir)
        g2ti.g2ti_masks(gdir)
        out = g2ti.distribute_thickness_vas(gdir, vas_c=0.034,
                                            dis_factor=0.2, topo_factor=0.2)

        ref = out * np.NaN

        i, j, _, _, thick = g2ti.get_ref_gtd_data(gdir)
        ref[j, i] = thick

        out = xr.DataArray(out)
        ref = xr.DataArray(ref)

        import matplotlib.pyplot as plt
        out.plot();
        plt.figure()
        ref.plot();
        plt.show()
