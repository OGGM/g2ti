import logging
import warnings
import os
import shutil
from distutils.version import LooseVersion

# External libs
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as shpg
import xarray as xr
import netCDF4
import salem
import rasterio
from scipy.interpolate import griddata
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import binary_erosion

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow
from oggm import entity_task
from oggm.core.gis import gaussian_blur

# Module logger
log = logging.getLogger(__name__)


def define_g2ti_glacier(path=None, base_dir=None):

    fname = os.path.join(path, 'outlines.shp')
    ent = gpd.read_file(fname)
    rid = ent.RGIId.values[0]
    if '5a' in rid:
        rid = rid.replace('5a', '60')

    ent['RGIId'] = rid
    gdir = utils.GlacierDirectory(ent.iloc[0], base_dir=base_dir)
    ent.to_file(gdir.get_filepath('outlines'))

    proj_out = salem.check_crs(ent.crs)

    # Also transform the intersects if necessary
    gdf = cfg.PARAMS['intersects_gdf']
    if len(gdf) > 0:
        gdf = gdf.loc[((gdf.RGIId_1 == gdir.rgi_id) |
                       (gdf.RGIId_2 == gdir.rgi_id))]
        if len(gdf) > 0:
            gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
            if hasattr(gdf.crs, 'srs'):
                # salem uses pyproj
                gdf.crs = gdf.crs.srs
            gdf.to_file(gdir.get_filepath('intersects'))
    else:
        # Sanity check
        if cfg.PARAMS['use_intersects']:
            raise RuntimeError('You seem to have forgotten to set the '
                               'intersects file for this run. OGGM works '
                               'better with such a file. If you know what '
                               'your are doing, set '
                               "cfg.PARAMS['use_intersects'] = False to "
                               "suppress this error.")

    # Topo
    shutil.copy(os.path.join(path, 'dem.tif'), gdir.get_filepath('dem'))
    mpath = gdir.get_filepath('dem').replace('dem', 'g2ti_mask')
    shutil.copy(os.path.join(path, 'mask.tif'), mpath)

    # Grid
    ds = salem.GeoTiff(gdir.get_filepath('dem'))
    ds.grid.to_json(gdir.get_filepath('glacier_grid'))
    gdir.write_pickle(['G2TI'], 'dem_source')

    return gdir


def parallel_define(paths):

    if cfg.PARAMS['use_multiprocessing']:
        mppool = workflow.init_mp_pool(cfg.CONFIG_MODIFIED)
        gdirs = mppool.map(define_g2ti_glacier, paths, chunksize=1)
    else:
        gdirs = [define_g2ti_glacier(p) for p in paths]
    return gdirs


@entity_task(log, writes=['gridded_data'])
def g2ti_masks(gdir):
    """Adds the g2ti mask to the netcdf file.

    Parameters
    ----------
    """

    # open srtm tif-file:
    dem_dr = rasterio.open(gdir.get_filepath('dem'), 'r', driver='GTiff')
    dem = dem_dr.read(1).astype(rasterio.float32)

    # Grid
    nx = dem_dr.width
    ny = dem_dr.height
    assert nx == gdir.grid.nx
    assert ny == gdir.grid.ny

    # Correct the DEM (ASTER...)
    # Currently we just do a linear interp -- ASTER is totally shit anyway
    min_z = -999.
    isfinite = np.isfinite(dem)
    if (np.min(dem) <= min_z) or np.any(~isfinite):
        xx, yy = gdir.grid.ij_coordinates
        pnan = np.nonzero((dem <= min_z) | (~isfinite))
        pok = np.nonzero((dem > min_z) | isfinite)
        points = np.array((np.ravel(yy[pok]), np.ravel(xx[pok]))).T
        inter = np.array((np.ravel(yy[pnan]), np.ravel(xx[pnan]))).T
        dem[pnan] = griddata(points, np.ravel(dem[pok]), inter)
        log.warning(gdir.rgi_id + ': DEM needed interpolation.')

    isfinite = np.isfinite(dem)
    if not np.all(isfinite):
        # see how many percent of the dem
        if np.sum(~isfinite) > (0.2 * nx * ny):
            raise RuntimeError('({}) too many NaNs in DEM'.format(gdir.rgi_id))
        log.warning('({}) DEM needed zeros somewhere.'.format(gdir.rgi_id))
        dem[isfinite] = 0

    if np.min(dem) == np.max(dem):
        raise RuntimeError('({}) min equal max in the DEM.'
                           .format(gdir.rgi_id))

    # Projection
    if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
        transf = dem_dr.transform
    else:
        transf = dem_dr.affine
    x0 = transf[2]  # UL corner
    y0 = transf[5]  # UL corner
    dx = transf[0]
    dy = transf[4]  # Negative

    if not (np.allclose(dx, -dy) or np.allclose(dx, gdir.grid.dx) or
            np.allclose(y0, gdir.grid.corner_grid.y0, atol=1e-2) or
            np.allclose(x0, gdir.grid.corner_grid.x0, atol=1e-2)):
        raise RuntimeError('DEM file and Salem Grid do not match!')
    dem_dr.close()

    # Clip topography to 0 m a.s.l.
    dem = dem.clip(0)

    # Smooth DEM?
    if cfg.PARAMS['smooth_window'] > 0.:
        gsize = np.rint(cfg.PARAMS['smooth_window'] / dx)
        smoothed_dem = gaussian_blur(dem, np.int(gsize))
    else:
        smoothed_dem = dem.copy()

    if not np.all(np.isfinite(smoothed_dem)):
        raise RuntimeError('({}) NaN in smoothed DEM'.format(gdir.rgi_id))

    with xr.open_rasterio(os.path.join(gdir.dir, 'g2ti_mask.tif')) as da:
        glacier_mask = np.squeeze(da.data)

    # Last sanity check based on the masked dem
    tmp_max = np.max(dem[np.where(glacier_mask == 1)])
    tmp_min = np.min(dem[np.where(glacier_mask == 1)])
    if tmp_max < (tmp_min + 1):
        raise RuntimeError('({}) min equal max in the masked DEM.'
                           .format(gdir.rgi_id))

    # write out the grids in the netcdf file
    nc = gdir.create_gridded_ncdf_file('gridded_data')

    v = nc.createVariable('topo', 'f4', ('y', 'x',), zlib=True)
    v.units = 'm'
    v.long_name = 'DEM topography'
    v[:] = dem

    v = nc.createVariable('topo_smoothed', 'f4', ('y', 'x',), zlib=True)
    v.units = 'm'
    v.long_name = ('DEM topography smoothed'
                   ' with radius: {:.1} m'.format(cfg.PARAMS['smooth_window']))
    v[:] = smoothed_dem

    v = nc.createVariable('glacier_mask', 'i1', ('y', 'x',), zlib=True)
    v.units = '-'
    v.long_name = 'Glacier mask'
    v[:] = glacier_mask

    # add some meta stats and close
    nc.max_h_dem = np.max(dem)
    nc.min_h_dem = np.min(dem)
    dem_on_g = dem[np.where(glacier_mask)]
    nc.max_h_glacier = np.max(dem_on_g)
    nc.min_h_glacier = np.min(dem_on_g)
    nc.close()


def get_ref_gtd_data(gdir):

    dd = '/home/mowglie/disk/G2TI/data/thickness_csv/'
    dd = dd + gdir.rgi_id.replace('RGI50', 'RGI60') + '_point_thickness.csv'
    dfp = pd.read_csv(dd, index_col=0)
    dfp = dfp[['lon', 'lat', 'thick']]
    ii, jj = gdir.grid.transform(dfp['lon'], dfp['lat'],
                                 crs=salem.wgs84,
                                 nearest=True)
    dfp['i'] = ii
    dfp['j'] = jj
    dfp['ij'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]
    dfp = dfp.groupby('ij').mean()
    return (dfp['i'].values, dfp['j'].values, dfp['lon'].values,
            dfp['lat'].values, dfp['thick'].values)


def _add_2d_interp_masks(gdir):
    """Computes the interpolation masks and adds them to the file."""

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    with netCDF4.Dataset(grids_file) as nc:
        topo = nc.variables['topo_smoothed'][:]
        glacier_mask = nc.variables['glacier_mask'][:]

    # Glacier exterior including nunataks
    erode = binary_erosion(glacier_mask)
    glacier_ext = glacier_mask ^ erode
    glacier_ext = np.where(glacier_mask == 1, glacier_ext, 0)

    # Intersects between glaciers
    gdfi = gpd.GeoDataFrame(columns=['geometry'])
    if gdir.has_file('intersects'):
        # read and transform to grid
        gdf = gpd.read_file(gdir.get_filepath('intersects'))
        salem.transform_geopandas(gdf, gdir.grid, inplace=True)
        gdfi = pd.concat([gdfi, gdf[['geometry']]])

    dx = gdir.grid.dx

    # Distance from border mask
    # Probably not the fastest way to do this, but it works
    dist = np.array([])
    jj, ii = np.where(glacier_ext)
    for j, i in zip(jj, ii):
        dist = np.append(dist, np.min(gdfi.distance(shpg.Point(i, j))))

    with warnings.catch_warnings():
        # https://github.com/Unidata/netcdf4-python/issues/766
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pok = np.where(dist <= 1)
    glacier_ext_intersect = glacier_ext * 0
    glacier_ext_intersect[jj[pok], ii[pok]] = 1

    # Scipy does the job
    dis_from_border = 1 + glacier_ext_intersect - glacier_ext
    dis_from_border = distance_transform_edt(dis_from_border) * dx
    dis_from_border[np.where(glacier_mask == 0)] = np.NaN

    # Slope mask
    sy, sx = np.gradient(topo, dx, dx)
    slope = np.arctan(np.sqrt(sy**2 + sx**2))
    slope = np.clip(slope, np.deg2rad(cfg.PARAMS['min_slope']*2), np.pi/2.)
    slope = 1 / slope**(cfg.N / (cfg.N+2))
    slope = np.where(glacier_mask == 1, slope, np.NaN)

    # Area above or below mask
    topo = np.where(glacier_mask == 1, topo, np.NaN)
    dis_from_minmax = topo * 0.
    jj, ii = np.nonzero(glacier_mask == 1)

    with warnings.catch_warnings():
        # https://github.com/Unidata/netcdf4-python/issues/766
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for j, i in zip(jj, ii):
            area_above = np.sum(topo > topo[j, i])
            area_below = np.sum(topo < topo[j, i])
            dis_from_minmax[j, i] = np.min([area_above, area_below])

    # smooth a little bit
    dis_from_minmax = dis_from_minmax**0.5

    assert not np.isfinite(slope[0, 0])
    assert not np.isfinite(dis_from_minmax[0, 0])
    assert not np.isfinite(dis_from_border[0, 0])    # write

    with netCDF4.Dataset(grids_file, 'a') as nc:

        vn = 'slope_mask'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f8', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Local slope (normalized)'
        v[:] = slope / np.nanmean(slope)

        vn = 'topo_mask'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f8', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Distance from top or bottom (normalized)'
        v[:] = dis_from_minmax / np.nanmean(dis_from_minmax)

        vn = 'distance_mask'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f8', ('y', 'x', ))
        v.units = '-'
        v.long_name = 'Distance from glacier border (normalized)'
        v[:] = dis_from_border / np.nanmean(dis_from_border)


@entity_task(log, writes=['gridded_data'])
def distribute_thickness_vas(gdir,
                             vas_c=None,
                             slope_factor=None,
                             dis_factor=None,
                             topo_factor=None,
                             input_filesuffix=''):
    """Compute a thickness map of the glacier using the nearest centerlines.

    This is a rather cosmetic task, not relevant for OGGM but for ITMIX.
    Here we take the nearest neighbors in a certain altitude range.

    Parameters
    ----------
    """

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    # See if we have the masks, else compute them
    with netCDF4.Dataset(grids_file) as nc:
        has_masks = 'slope_mask' in nc.variables
    if not has_masks:
        _add_2d_interp_masks(gdir)

    # Check input
    if slope_factor is None and topo_factor is None and dis_factor is None:
        slope_factor = 1/3

    # One is set
    if topo_factor is None and dis_factor is None:
        topo_factor = (1 - slope_factor) / 2
        dis_factor = (1 - slope_factor) / 2
    if slope_factor is None and dis_factor is None:
        slope_factor = (1 - topo_factor) / 2
        dis_factor = (1 - topo_factor) / 2
    if slope_factor is None and topo_factor is None:
        topo_factor = (1 - dis_factor) / 2
        slope_factor = (1 - dis_factor) / 2

    # Two are set
    if slope_factor is None:
        slope_factor = 1 - dis_factor - topo_factor
    if topo_factor is None:
        topo_factor = 1 - dis_factor - slope_factor
    if dis_factor is None:
        dis_factor = 1 - slope_factor - topo_factor

    # Make sure they sum to one
    slope_factor = np.clip(slope_factor, 0, 1)
    dis_factor = np.clip(dis_factor, 0, 1)
    topo_factor = np.clip(topo_factor, 0, 1)
    f = 1 / np.sum([slope_factor, dis_factor, topo_factor])

    # Variables
    grids_file = gdir.get_filepath('gridded_data')
    with netCDF4.Dataset(grids_file) as nc:
        with warnings.catch_warnings():
            # https://github.com/Unidata/netcdf4-python/issues/766
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            slope = nc.variables['slope_mask'][:]
            dis_from_minmax = nc.variables['topo_mask'][:]
            dis_from_border = nc.variables['distance_mask'][:]
        glacier_mask = nc.variables['glacier_mask'][:]

    # Final normalizing
    slope *= slope_factor * f
    dis_from_minmax *= topo_factor * f
    dis_from_border *= dis_factor * f
    test = (np.nanmean(slope) +
            np.nanmean(dis_from_minmax) +
            np.nanmean(dis_from_border))
    np.testing.assert_allclose(test, 1, atol=1e-2,
                               err_msg='Something went wrong with the '
                                       'factor normalization')

    # Original VAS volume in m3
    if vas_c is None:
        vas_c = 0.034
    inv_vol = vas_c * (gdir.rgi_area_km2 ** 1.375)
    inv_vol *= 1e9

    # Naive thickness
    dx = gdir.grid.dx
    thick = glacier_mask * inv_vol

    # Times each factor (inefficient but less sensitive to float error)
    thick = thick * slope + thick * dis_from_minmax + thick * dis_from_border

    # Conserve volume
    thick *= inv_vol / np.nansum(thick * dx**2)

    # Write
    with netCDF4.Dataset(grids_file, 'a') as nc:
        vn = 'distributed_thickness'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f8', ('y', 'x', ))
        v.units = 'm'
        v.long_name = 'Local ice thickness'
        v[:] = thick

    return thick


@entity_task(log, writes=['gridded_data'])
def optimize_distribute_thickness_single_glacier(gdir):
    """Tests many things

    Parameters
    ----------
    """

    # Get the ref data
    i, j, lon, lat, ref_thick = get_ref_gtd_data(gdir)
    grids_file = gdir.get_filepath('gridded_data')
    with netCDF4.Dataset(grids_file) as nc:
        glacier_mask = nc.variables['glacier_mask'][:]
    pok = np.nonzero(glacier_mask[j, i])
    assert len(pok[0]) > 0
    i, j, ref_thick = i[pok], j[pok], ref_thick[pok]

    # Make the parameter space
    fac_slope = np.linspace(0, 1, 11)
    fac_dis = np.linspace(0, 1, 11)
    fac_c = 0.027 + np.arange(13) * 0.001

    # Make the point matrix
    out_mat = np.full((len(ref_thick), len(fac_c),
                       len(fac_slope), len(fac_dis)),
                      np.NaN)

    out = pd.DataFrame()
    t = 0
    for zi, fc in enumerate(fac_c):
        for yi, fs in enumerate(fac_slope):
            for xi, fd in enumerate(fac_dis):
                if (fs + fd) > 1:
                    continue
                thick = distribute_thickness_vas(gdir, reset=True,
                                                 vas_c=fc,
                                                 slope_factor=fs,
                                                 dis_factor=fd,
                                                 print_log=False)
                thick = thick[j, i]
                assert np.all(np.isfinite(thick))
                out.loc[t, 'vas_c'] = fc
                out.loc[t, 'fac_slope'] = fs
                out.loc[t, 'fac_dis'] = fd
                out.loc[t, 'fac_topo'] = 1 - fs - fd
                out.loc[t, 'bias'] = np.mean(thick[pok]-ref_thick[pok])
                out.loc[t, 'mad'] = utils.mad(ref_thick[pok], thick[pok])
                out.loc[t, 'rmsd'] = utils.rmsd(ref_thick[pok], thick[pok])
                out.loc[t, 'n_ref'] = len(ref_thick[pok])
                out_mat[:, zi, yi, xi] = thick
                t += 1

    fs = os.path.join(gdir.dir, 'distribute_optim.csv')
    out.to_csv(fs)

    fs = os.path.join(gdir.dir, 'point_thick.nc')
    dims = ['points', 'fac_c', 'fac_slope', 'fac_dis']
    coords = {'points':np.arange(len(ref_thick)),
              'fac_c':fac_c,
              'fac_slope':fac_slope,
              'fac_dis':fac_dis}
    out_mat = xr.DataArray(out_mat, dims=dims, coords=coords)
    out_mat = out_mat.to_dataset(name='oggm')
    out_mat['ref_thick'] = (('points',), ref_thick)
    out_mat.to_netcdf(fs)
    return out, out_mat
