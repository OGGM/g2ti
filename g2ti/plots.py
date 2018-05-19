import netCDF4
import numpy as np
from matplotlib import cm as colormap
import salem
from oggm.graphics import _plot_map, truncate_colormap
from g2ti import tasks as g2tasks
from oggm.utils import entity_task


@_plot_map
def plot_domain_with_gtd(gdirs, ax=None, smap=None):
    """Plot the glacier directory."""

    # Files
    gdir = gdirs[0]
    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    try:
        smap.set_data(topo)
    except ValueError:
        pass

    cm = truncate_colormap(colormap.terrain, minval=0.25, maxval=1.0, n=256)
    smap.set_cmap(cm)
    smap.set_plot_params(nlevels=256)

    try:
        for gdir in gdirs:
            crs = gdir.grid.center_grid
            geom = gdir.read_pickle('geometries')

            # Plot boundaries
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='white',
                                  alpha=0.3, zorder=2, linewidth=.2)
            for l in poly_pix.interiors:
                smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
    except FileNotFoundError:
        smap.set_shapefile(gdir.get_filepath('outlines'), color='black')

    smap.plot(ax)

    # Get the ref data
    i, j, lon, lat, ref_thick = g2tasks.get_ref_gtd_data(gdir)
    grids_file = gdir.get_filepath('gridded_data')
    with netCDF4.Dataset(grids_file) as nc:
        glacier_mask = nc.variables['glacier_mask'][:]
    pok = np.nonzero(glacier_mask[j, i])
    assert len(pok[0]) > 0
    i, j, ref_thick = i[pok], j[pok], ref_thick[pok]
    dl = salem.DataLevels(ref_thick)
    ax.scatter(i, j, color=dl.to_rgb(), s=50, edgecolors='k', linewidths=1)
    tstr = ' (N: {})'.format(len(ref_thick))

    return dict(cbar_label='Alt. [m]', title_comment=tstr)