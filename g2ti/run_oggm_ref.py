# Python imports
from os import path
import glob
import oggm
import pandas as pd

# Module logger
import logging
log = logging.getLogger(__name__)

# Libs
import salem

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
import g2ti

# For timing the run
import time
start = time.time()

version = '61'

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True

# test directory
idir = g2ti.geometry_dir
cfg.PATHS['working_dir'] = '/home/mowglie/disk/G2TI/oggm_ref_run'
utils.mkdir(cfg.PATHS['working_dir'])

fl = glob.glob('/home/mowglie/disk/G2TI/plots_visual_sel/*.png')
sel_ids = [f.split('_')[-1].replace('.png', '') for f in fl if
           ('-19' not in f) and ('-05' not in f)]

# Tests only
sel_ids = sel_ids[0:3]

rgidf = utils.get_rgi_glacier_entities(sel_ids, version=version)

# Add intersects
db = utils.get_rgi_intersects_region_file(version='61', rgi_ids=sel_ids)
cfg.set_intersects_db(db)

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)


log.info('Starting OGGM run')
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)

# Preprocessing tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,
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
# We use the default parameters for this run
execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    varname_suffix='_alt_0.5_smooth_d')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    smooth_radius=1,
                    varname_suffix='_alt_0.5_smooth_1')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    smooth_radius=5,
                    varname_suffix='_alt_0.5_smooth_5')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    smooth_radius=10,
                    varname_suffix='_alt_0.5_smooth_10')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    dis_from_border_exp=1.,
                    varname_suffix='_alt_1.0_smooth_d')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    dis_from_border_exp=0.1,
                    varname_suffix='_alt_0.1_smooth_d')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    dis_from_border_exp=0.2,
                    varname_suffix='_alt_0.2_smooth_d')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    smooth_radius=5,
                    dis_from_border_exp=1.,
                    varname_suffix='_alt_1.0_smooth_5')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    smooth_radius=5,
                    dis_from_border_exp=0.1,
                    varname_suffix='_alt_0.1_smooth_5')

execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                    smooth_radius=5,
                    dis_from_border_exp=0.2,
                    varname_suffix='_alt_0.2_smooth_5')

execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                    varname_suffix='_int_smooth_d')

execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                    smooth_radius=1,
                    varname_suffix='_int_smooth_1')

execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                    smooth_radius=5,
                    varname_suffix='_int_smooth_5')

execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                    smooth_radius=10,
                    varname_suffix='_int_smooth_10')

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
