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
from oggm.sandbox import g2ti

# For timing the run
import time
start = time.time()

version = '61'

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False

# test directory
idir = '/home/mowglie/disk/G2TI/data/geometry/'
bdir = '/home/mowglie/disk/G2TI/vas_run'
cfg.PATHS['working_dir'] = bdir
utils.mkdir(bdir, reset=True)

df = pd.read_csv('/home/mowglie/Documents/g2ti/GTD/INDEX_thickness_data.csv')
df = df.loc[df.n_pts > 0]

rgi_ids = df.RGIId.values
flist = [idir + r[:8] + '/' + r for r in rgi_ids]

cfg.set_intersects_db(utils.get_rgi_intersects_region_file(version=version,
                                                           rgi_ids=rgi_ids))

gdirs = g2ti.parallel_define(flist)

# Preprocessing tasks
task_list = [
    g2ti.g2ti_masks,
    g2ti.optimize_distribute_thickness_single_glacier,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
