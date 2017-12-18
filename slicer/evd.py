from subprocess import call
from datetime import datetime
from time import sleep
import argparse
import pyautogui
import cv2
import numpy as np


x0 = 6
y0 = 132
width = 1247
height = 746


def scan(figure_dir, prefix):
    for i in range(100):
        time = datetime.now().strftime('%Y_%m_%d.%H_%M_%S')
        print(time)
        call('screencapture -x -R{},{},{},{} {}/{}{}.png'.format(x0, y0, width, height, figure_dir, prefix, time), shell=True)
        sleep(10)
    print('Finished.')


def single(figure_dir, **kwargs):
    prefix = kwargs.get('prefix', '')
    suffix = kwargs.get('suffix', datetime.now().strftime('%Y_%m_%d.%H_%M_%S'))
    print(suffix)
    call('screencapture -x -R{},{},{},{} {}/{}{}{}.png'.format(x0, y0, width, height, figure_dir, prefix, '.' if prefix != '' else '', suffix), shell=True)


def click():
    print(pyautogui.position())
    pyautogui.moveTo(115, 84)
    pyautogui.click()


def single_click(figure_dir, prefix):
    pyautogui.hotkey('command', 'tab')
    click()
    for i in range(200):
        single(figure_dir, prefix=prefix, suffix=str(i))
        click()
        sleep(15)


def join(figure_dir, prefix):
    for i in range(200):
        print('processing event ', i)
        raw = cv2.imread('{}/{}.{}.png'.format(figure_dir, prefix, i))
        slicer4d = cv2.imread('{}/{}.slicer4d.{}.png'.format(figure_dir, prefix, i))
        tdslicer_2d = cv2.imread('{}/{}.tdslicer.2d.{}.png'.format(figure_dir, prefix, i))
        tdslicer_merge = cv2.imread('{}/{}.tdslicer.merge.{}.png'.format(figure_dir, prefix, i))

        top = np.concatenate((raw, slicer4d), axis=1)
        bottom = np.concatenate((tdslicer_2d, tdslicer_merge), axis=1)
        full = np.concatenate((top, bottom), axis=0)

        cv2.putText(full, 'raw', (5, 620), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(full, 'Slicer4D', (1250, 620), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(full, 'TDSlicer (2D)', (5, 1365), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(full, 'TDSlicer (merge)', (1250, 1365), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imwrite('{}/{}.all.{}.png'.format(figure_dir, prefix, i), full)
        # cv2.imwrite('test.png', full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--figure_dir', help='figure dir', default='.')
    parser.add_argument('-p', '--prefix', help='figure name prefix', default='')
    args = parser.parse_args()

    # pyautogui.hotkey('command', 'tab')
    # click()
    # pyautogui.hotkey('command', 'tab')
    # single(args.figure_dir, prefix=args.prefix)
    # single_click(args.figure_dir, prefix=args.prefix)
    join(args.figure_dir, prefix=args.prefix)
