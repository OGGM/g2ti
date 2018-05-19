import os
# This is hard coded version string.
# Real packages use more sophisticated methods to make sure that this
# string is synchronised with `setup.py`, but for our purposes this is OK
__version__ = '0.0.1'

from configobj import ConfigObj, ConfigObjError

cp = ConfigObj(os.path.expanduser('~/.g2ti_paths'), file_error=True)

# Paths
geometry_dir = cp['geometry_dir']
index_file = cp['index_file']
thickness_csv_dir = cp['thickness_csv_dir']
index_file_orig = cp['index_file_orig']
