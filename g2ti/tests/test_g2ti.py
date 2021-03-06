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
import salem
import geopandas as gpd

import g2ti
from g2ti import tasks as g2task
from g2ti import plots

do_plot = False


class TestG2TI(unittest.TestCase):

    # Test case for a tidewater glacier

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_g2ti')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        self.idir = os.path.join(g2ti.geometry_dir, 'RGI60-11',
                                 'RGI60-11.00887')

        # Init
        cfg.initialize()
        cfg.PATHS['working_dir'] = self.testdir
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
        gdir = g2task.define_g2ti_glacier(self.idir, base_dir=self.testdir)
        g2task.g2ti_masks(gdir)
        if do_plot:
            plots.plot_domain_with_gtd(gdir)
            import matplotlib.pyplot as plt
            plt.show()

    def test_g2ti_data(self):
        gdir = g2task.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        g2task.g2ti_masks(gdir)

        i, j, _, _, thick = g2task.get_ref_gtd_data(gdir)

        ds = xr.open_dataset(gdir.get_filepath('gridded_data'))

        is_mask = ds.glacier_mask.isel_points(x=i, y=j)
        assert np.all(is_mask)

    def test_optim(self):
        gdir = g2task.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        g2task.g2ti_masks(gdir)
        out, ds = g2task.optimize_distribute_thickness_single_glacier(gdir)

        np.testing.assert_allclose(out.bias.abs().min(), 0, atol=0.5)

        out_s = out.loc[out['bias'].abs() < 5]
        out_s = out_s.loc[out_s.idxmin()['mad']]

        bias = ds['thick'] - ds['ref_thick']
        mbias = np.abs(bias.mean(dim='points'))
        mbias = mbias.min(dim=['fac_dis', 'fac_slope'])
        assert mbias.min() < 1

        g2task.merge_point_data([gdir, gdir, gdir])
        outf = os.path.join(cfg.PATHS['working_dir'], 'point_thick.nc')
        ds = xr.open_dataset(outf)
        print(ds)

        if do_plot:
            import matplotlib.pyplot as plt
            ds.ref_thick.plot();
            plt.figure()
            ds['experiments'].data = np.arange(len(ds['experiments']))
            ds.thick.plot()
            plt.show()

    def test_distribute(self):
        gdir = g2task.define_g2ti_glacier(self.idir, base_dir=self.testdir)

        gis.glacier_masks(gdir)
        g2task.g2ti_masks(gdir)
        out = g2task.distribute_thickness_vas(gdir, vas_c=0.034,
                                              dis_factor=0.2, topo_factor=0.2,
                                              write_tiff=True)

        ref = out * np.NaN

        i, j, _, _, thick = g2task.get_ref_gtd_data(gdir)
        ref[j, i] = thick

        out = xr.DataArray(out)
        ref = xr.DataArray(ref)
        ft = os.path.join(cfg.PATHS['working_dir'], 'final',
                          'RGI60-{}'.format(gdir.rgi_region))
        ft = os.path.join(ft, 'thickness_{}.tif'.format(gdir.rgi_id))
        tif = xr.open_rasterio(ft)

        dx2 = salem.GeoTiff(ft).grid.dx**2

        np.testing.assert_allclose((tif * dx2).sum()*1e-9,
                                   0.034*gdir.rgi_area_km2**1.375,
                                   rtol=0.01)

        if do_plot:
            import matplotlib.pyplot as plt
            out.plot();
            plt.figure()
            ref.plot();
            plt.figure()
            tif.plot();
            plt.show()


class TestOGGMtoTIFF(unittest.TestCase):

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['working_dir'] = self.testdir

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_to_tiff(self):

        from oggm import tasks
        from oggm.workflow import execute_entity_task
        from oggm import GlacierDirectory
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = GlacierDirectory(entity, base_dir=self.testdir)
        gdir.rgi_id = gdir.rgi_id.replace('50-', '60-')
        gis.define_glacier_region(gdir, entity=entity)
        gdirs = [gdir]

        # Preprocessing tasks
        task_list = [
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Climate tasks -- only data IO and tstar interpolation!
        execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.distribute_t_stars(gdirs)
        execute_entity_task(tasks.apparent_mb, gdirs)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        execute_entity_task(tasks.volume_inversion, gdirs,
                            glen_a=cfg.A*3,
                            fs=0)
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

        g2task.oggm_to_g2ti(gdir)

        ft = os.path.join(cfg.PATHS['working_dir'], 'final',
                          'RGI60-{}'.format(gdir.rgi_region))
        ft = os.path.join(ft, 'thickness_{}.tif'.format(gdir.rgi_id))

        ds = xr.open_rasterio(ft)

        tpl_f = os.path.join(g2ti.geometry_dir, gdir.rgi_id[:8], gdir.rgi_id,
                             'mask.tif')
        da = xr.open_rasterio(tpl_f)

        np.testing.assert_allclose(ds.sum(), (da*ds).sum(), rtol=0.05)

        if do_plot:
            import matplotlib.pyplot as plt
            ds.plot()
            plt.figure()
            da.plot()
            plt.show()
