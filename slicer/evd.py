from subprocess import call
from datetime import datetime
from time import sleep
import argparse
import pyautogui


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
    for i in range(200):
        single(figure_dir, prefix=prefix, suffix=str(i))
        click()
        sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--figure_dir', help='figure dir', default='.')
    parser.add_argument('-p', '--prefix', help='figure name prefix', default='')
    args = parser.parse_args()

    # pyautogui.hotkey('command', 'tab')
    # click()
    # pyautogui.hotkey('command', 'tab')
    # single(args.figure_dir, prefix=args.prefix)
    single_click(args.figure_dir, prefix=args.prefix)
