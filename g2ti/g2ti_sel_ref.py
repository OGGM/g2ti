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
from g2ti import tasks as g2tasks

# For timing the run
import time
start = time.time()

version = '61'

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False

# test directory
idir = g2ti.geometry_dir
cfg.PATHS['working_dir'] = '/home/mowglie/disk/G2TI/vas_plot'
utils.mkdir(cfg.PATHS['working_dir'], reset=True)

df = pd.read_csv(g2ti.index_file)
df = df.loc[df.n_pts > 49]  # 582 glaciers

rgi_ids = df.RGIId.values
flist = [idir + r[:8] + '/' + r for r in rgi_ids]

cfg.set_intersects_db(utils.get_rgi_intersects_region_file(version=version,
                                                           rgi_ids=rgi_ids))
gdirs = g2tasks.parallel_define(flist)

# Preprocessing tasks
task_list = [
    g2tasks.g2ti_masks,
]
for task in task_list:
    execute_entity_task(task, gdirs)

from g2ti import plots
from functools import partial
func = partial(plots.plot_domain_with_gtd, autosave=True)
execute_entity_task(func, gdirs)


# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
