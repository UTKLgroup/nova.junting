from rootalias import *
import math
from math import cos, pi


FIGURE_DIR = 'figures'


def convert_rgb_color(rgbs):
    print(','.join(['{:.2f}'.format(rgb / 255.) for rgb in rgbs]))


def calculate_component_position():
    # scraper 1
    # theta = 0.65
    # angle_correction_factor = cos(theta * pi / 180.)
    # print('angle_correction_factor = {}'.format(angle_correction_factor))
    # print(445.9224 + (127.2 / 2. + 914.4 / 2.))
    # print(445.9224 - (127.2 / 2. + 1828.8 / 2.))
    # print(445.9224 + (127.2 / 2. + 914.4 / 2.) / angle_correction_factor)
    # print(445.9224 - (127.2 / 2. + 1828.8 / 2.) / angle_correction_factor)

    # scraper 2
    # theta = 0.66
    # angle_correction_factor = cos(theta * pi / 180.)
    # print(583.692 + (50.8 / 2. + 914.4 / 2.) / angle_correction_factor)
    # print(583.692 - (50.8 / 2. + 1828.8 / 2.) / angle_correction_factor)
    # print((20064.4 - 3047. / 2.) - (16711.9 + 3047. / 2.))

    # vacuum
    # print(4908.69 - 141.29 / 2.)

    # Coll_All, upstream
    # print(71.95 + (31.75 / 2. + 914.4 / 2.) / cos(0.99 * pi / 180.))
    # print(71.95 - (31.75 / 2. + 914.4 / 2.) / cos(0.99 * pi / 180.))

    # cage
    print(24.1 + (450. / 2. + 100. / 2.) / cos(1.31 * pi / 180.))
    print(24.1 - (450. / 2. + 100. / 2.) / cos(1.31 * pi / 180.))
    print(450. * (1. - 1. / cos(1.31 * pi / 180.)))


def plot_particle_position():
    tf = TFile('LAPPD_MC_1g.root')

    detectors = ['CollUpstream', 'Scraper1', 'Scraper2', 'CollDownstream', 'NovaTarget']
    h_xy = TH2D('h_xy', 'h_xy', 120, -3, 3, 120, -2, 4)
    h_pxpy = TH2D('h_pxpy', 'h_pxpy', 40, -4, 4, 40, -4, 4)

    for particle in tf.Get('VirtualDetector/{}'.format(detectors[4])):
        # print(int(particle.PDGid))
        # if int(particle.PDGid) == 2212 or 211:
        # if int(particle.PDGid) == 211:
        # if int(particle.PDGid) == 2212:
        # if int(particle.PDGid) == -13:
        h_xy.Fill(particle.x / 1000., particle.y / 1000.)
        h_pxpy.Fill(particle.Px / 1000., particle.Py / 1000.)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetRightMargin(0.15)
    gPad.SetLogz()

    set_h2_color_style()
    # set_h2_style(h_xy)
    # h_xy.Draw('colz')

    set_h2_style(h_pxpy)
    h_pxpy.Draw('colz')

    c1.Update()
    c1.SaveAs('{}/plot_particle_position.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


# 20180331_secondary_beam
gStyle.SetOptStat(0)
plot_particle_position()
# calculate_component_position()
# convert_rgb_color([201, 96, 35]) # quadrupole, color based on native values from mac color meter
# convert_rgb_color([9, 78, 149])  # dipole
# convert_rgb_color([78, 139, 50]) # trim
# convert_rgb_color([0, 204, 204]) # collimator
# convert_rgb_color([204, 0, 0])   # scraper
