from subprocess import call
from time import sleep
import os.path


def run_genie():
    concatenate_filename = 'gntp.1000.ghep.root.txt'
    if os.path.isfile(concatenate_filename):
        call('mv -f {} tmp/'.format(concatenate_filename), shell=True)

    for i in range(1, 17):
        print('processing m = ', i)
        prefix = 'm_{}'.format(i)
        call('gevgen_nosc -m {} -g 1000060120 --seed 1 -o {} -n 1'.format(i, prefix), shell=True)
        root_filename = '{}.1000.ghep.root'.format(prefix, i)
        call('./gtestEventLoop -f {}'.format(root_filename), shell=True)
        call('cat {}.txt >> {}'.format(root_filename, concatenate_filename), shell=True)


run_genie()
