import os
from subprocess import call
from util import *


dir_scratch = '/pnfs/nova/scratch/users/junting/slicer'
dir_cp = '/nova/app/users/junting/slicer/data/slicer'

delete_empty_file(dir_cp, '/nova/app/users/junting/slicer/data/tmp')

# with open('tmp/scratch.txt') as f_list:
#     scratch_files = []
#     for row in f_list.readlines():
#         row = row.strip()
#         if not os.path.isfile('{}/{}'.format(dir_cp, os.path.basename(row))):
#             scratch_files.append(row)
#     print('Copying {} files.'.format(len(scratch_files)))

#     for sub_scratch_files in split_list(scratch_files, 100):
#         print(len(sub_scratch_files))
#         cmd = 'ifdh cp -D {} {}/'.format(' '.join(sub_scratch_files), dir_cp)
#         call(cmd, shell=True)
