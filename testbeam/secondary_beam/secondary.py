
def convert_rgb_color(rgbs):
    print(','.join(['{:.2f}'.format(rgb / 255.) for rgb in rgbs]))

# 20180331_secondary_beam
convert_rgb_color([201, 96, 35]) # quadrupole, color based on native values from mac color meter
convert_rgb_color([9, 78, 149])  # dipole
convert_rgb_color([78, 139, 50]) # trim
convert_rgb_color([0, 204, 204]) # collimator
convert_rgb_color([204, 0, 0]) # scraper
