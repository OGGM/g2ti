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
from g2ti import tasks as g2tasks

# For timing the run
import time
start = time.time()

version = '61'
region = '11'

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False

# Working directory
cfg.PATHS['working_dir'] = '/home/mowglie/disk/G2TI/vas_region'

geom_dir = g2ti.geometry_dir
flist = glob.glob(os.path.join(geom_dir, 'RGI60-{}'.format(region), '*'))

# For testing
flist = flist[:10]

cfg.set_intersects_db(utils.get_rgi_intersects_region_file(region=region,
                                                           version=version))
gdirs = g2tasks.parallel_define(flist)

# Preprocessing tasks
execute_entity_task(g2tasks.g2ti_masks, gdirs)
execute_entity_task(g2tasks.distribute_thickness_vas, gdirs,
                    vas_c=0.034,
                    slope_factor=0.7,
                    dis_factor=0.2,
                    write_tiff=True)

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
