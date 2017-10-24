from subprocess import call
from datetime import datetime
from time import sleep

x0 = 6
y0 = 131
width = 1247
height = 747
figure_dir = 'data/evd/fd_data_cosmic'
prefix = 'fd_data_cosmic'

for i in range(100):
    time = datetime.now().strftime('%Y_%m_%d.%H_%M_%S')
    print(time)
    call('screencapture -x -R{},{},{},{} {}/{}.{}.png'.format(x0, y0, width, height, figure_dir, prefix, time), shell=True)
    sleep(10)

print('Finished.')
