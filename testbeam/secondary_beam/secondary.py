import math
from math import cos, pi

def convert_rgb_color(rgbs):
    print(','.join(['{:.2f}'.format(rgb / 255.) for rgb in rgbs]))


def calculate_scraper_y_position():
    theta = 0.65
    # theta = 0.66
    angle_correction_factor = cos(theta * pi / 180.)
    print('angle_correction_factor = {}'.format(angle_correction_factor))

    # scraper 1
    print(445.9224 + (127.2 / 2. + 914.4 / 2.))
    print(445.9224 - (127.2 / 2. + 1828.8 / 2.))
    # print(445.9224 + (127.2 / 2. + 914.4 / 2.) / angle_correction_factor)
    # print(445.9224 - (127.2 / 2. + 1828.8 / 2.) / angle_correction_factor)

    # scraper 2
    # print(583.692 + (50.8 / 2. + 914.4 / 2.) / angle_correction_factor)
    # print(583.692 - (50.8 / 2. + 1828.8 / 2.) / angle_correction_factor)


# 20180331_secondary_beam
calculate_scraper_y_position()
# convert_rgb_color([201, 96, 35]) # quadrupole, color based on native values from mac color meter
# convert_rgb_color([9, 78, 149])  # dipole
# convert_rgb_color([78, 139, 50]) # trim
# convert_rgb_color([0, 204, 204]) # collimator
# convert_rgb_color([204, 0, 0])   # scraper
