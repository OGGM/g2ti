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
factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
factors += [1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
factors += [6, 7, 8, 9, 10]

smooth_rad = [None, 1, 3, 5, 10]

dis_expos = [0.1, 0.25, 0.5, 1.]

# We use the default parameters for this run
factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
factors += [1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
factors += [6, 7, 8, 9, 10]

smooth_rad = [None, 5, 10]
dis_expos = [0.25, 0.5, 1.]

suffix = []
for ga in factors:
    execute_entity_task(tasks.volume_inversion, gdirs,
                        glen_a=cfg.A * ga, fs=0)

    for sr in smooth_rad:
        _sr = 0 if sr is None else sr
        suffix = 'A{:02.1f}_S{:02d}'.format(ga, _sr)

        execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                            varname_suffix='_int_'+suffix)

        for de in dis_expos:
            _sr = 0 if sr is None else sr
            suffix = 'A{:02.1f}_S{:02d}_D{:1.2f}'.format(ga, _sr, de)

            execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                                smooth_radius=sr,
                                dis_from_border_exp=de,
                                varname_suffix='_alt_' + suffix)

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
