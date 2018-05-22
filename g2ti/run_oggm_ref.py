# Python imports
import os
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

df = pd.read_csv(g2ti.index_file, index_col=0)
sel_ids = [rid for rid in df.index if ('-19.' not in rid) and ('-05.' not in rid)]

sel_ids = sel_ids[:3]

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
a_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
a_factors += [1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
a_factors += [6, 7, 8, 9, 10]

s_factors = [0., 0.2, 0.5]

suffix = []
for fa in a_factors:
    for fs in s_factors:
        execute_entity_task(tasks.volume_inversion, gdirs,
                            glen_a=cfg.A * fa, fs=cfg.FS * fs)

        suffix = '_A{:04.1f}_S{:03.1}'.format(fa, fs)

        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                            varname_suffix=suffix)

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
