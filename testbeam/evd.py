from subprocess import call
from datetime import datetime
from time import sleep
import argparse
import pyautogui
import cv2
import numpy as np


def position():
    print(pyautogui.position())
    pyautogui.moveTo(80, 260)
    pyautogui.click()
    pyautogui.moveTo(750, 570)
    pyautogui.click()
    pyautogui.dragTo(1501, 950, button='left')
    # pyautogui.hotkey('command', 'shift', '5')
    # pyautogui.moveTo(180, 420)
    # pyautogui.mouseDown()
    # pyautogui.drag(1070, 240, 1, button='left')


def take_screen_shot(figure_dir, prefix, timestamp):
    filename = 'evd.'
    if prefix:
        filename += prefix + '.'
    if timestamp:
        filename += datetime.now().strftime('%Y%m%d_%H%M%S.')
    filename += 'pdf'
    x0 = 180
    y0 = 420
    width = 1130
    height = 250
    call('screencapture -x -t pdf -R{},{},{},{} {}/{}'.format(x0, y0, width, height, figure_dir, filename), shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--figure_dir', help='figure dir', default='.')
    parser.add_argument('-p', '--prefix', help='figure name prefix', default='')
    parser.add_argument('-t', '--timestamp', help='add timestamp in filename', action='store_true')
    args = parser.parse_args()

    position()
    take_screen_shot(args.figure_dir, args.prefix, args.timestamp)
