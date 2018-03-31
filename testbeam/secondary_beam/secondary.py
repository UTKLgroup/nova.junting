
def convert_rgb_color(rgbs):
    print([float('{:.2f}'.format(rgb / 255.)) for rgb in rgbs])

# 20180331_secondary_beam
convert_rgb_color([201, 96, 35]) # quadrupole
convert_rgb_color([9, 78, 149])  # dipole
convert_rgb_color([78, 139, 50]) # trim
