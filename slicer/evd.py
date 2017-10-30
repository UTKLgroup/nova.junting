from subprocess import call
from datetime import datetime
from time import sleep
import argparse


x0 = 6
y0 = 131
width = 1247
height = 747


def scan(figure_dir, prefix):
    for i in range(100):
        time = datetime.now().strftime('%Y_%m_%d.%H_%M_%S')
        print(time)
        call('screencapture -x -R{},{},{},{} {}/{}{}.png'.format(x0, y0, width, height, figure_dir, prefix, time), shell=True)
        sleep(10)
    print('Finished.')


def single(figure_dir, prefix):
    time = datetime.now().strftime('%Y_%m_%d.%H_%M_%S')
    call('screencapture -x -R{},{},{},{} {}/{}{}.png'.format(x0, y0, width, height, figure_dir, prefix, time), shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--figure_dir', help='figure dir', default='.')
    parser.add_argument('-p', '--prefix', help='figure name prefix', default='')
    args = parser.parse_args()

    prefix = args.prefix if args.prefix.endswith('.') else args.prefix + '.'
    single(args.figure_dir, args.prefix)
