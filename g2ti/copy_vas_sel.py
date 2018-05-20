import pandas as pd
import shutil
import os
from g2ti import index_file_orig
from oggm.utils import mkdir

dfexp = pd.read_csv(index_file_orig, skiprows=18, delim_whitespace=True,
                    index_col=1)

mkdir('cali1')
mkdir('cali2')
mkdir('cali3')

# thickness_RGI60-06.00041.tif
for rid in dfexp.index:
    reg = 'RGI60-{}'.format(rid[6:8])
    fi = 'thickness_{}.tif'.format(rid)
    f = os.path.join('final', reg, fi)

    of = os.path.join('cali1', fi.replace('.tif', '_c1.tif'))
    shutil.copyfile(f, of)
    of = os.path.join('cali2', fi.replace('.tif', '_c2.tif'))
    shutil.copyfile(f, of)
    of = os.path.join('cali3', fi.replace('.tif', '_c3.tif'))
    shutil.copyfile(f, of)
