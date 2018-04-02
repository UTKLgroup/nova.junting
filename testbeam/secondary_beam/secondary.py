from rootalias import *
import math
from math import cos, pi


FIGURE_DIR = '/Users/juntinghuang/beamer/20180331_secondary_beam/figures'


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


def plot_xy():
    tf = TFile('LAPPD_MC_1g.1m.root')

    detectors = ['CollUpstream', 'Scraper1', 'Scraper2', 'D3', 'CollDownstream', 'NovaTarget']
    h_xys = []

    for detector in detectors:
        print('detector = {}'.format(detector))
        h_xy = TH2D('h_{}'.format(detector), 'h_{}'.format(detector), 120, -3, 3, 120, -2, 4)
        set_h2_style(h_xy)
        h_xys.append(h_xy)
        for particle in tf.Get('VirtualDetector/{}'.format(detector)):
            h_xy.Fill(particle.x / 1000., particle.y / 1000.)

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetRightMargin(0.18)
    gPad.SetLogz()
    set_h2_color_style()

    label_title_size = 32
    tex = TLatex()
    tex.SetTextFont(63)
    tex.SetTextSize(30)
    tex.SetTextAlign(33)
    tex.SetTextColor(kRed)

    for i, h_xy in enumerate(h_xys):
        h_xy.GetXaxis().SetTitle('X (m)')
        h_xy.GetYaxis().SetTitle('Y (m)')
        h_xy.GetXaxis().SetLabelSize(label_title_size)
        h_xy.GetYaxis().SetLabelSize(label_title_size)
        h_xy.GetZaxis().SetLabelSize(label_title_size)
        h_xy.GetXaxis().SetTitleSize(label_title_size)
        h_xy.GetYaxis().SetTitleSize(label_title_size)
        h_xy.GetZaxis().SetTitleSize(label_title_size)
        h_xy.Draw('colz')
        tex.DrawLatex(2.7, 3.7, 'Detector {}'.format(i + 1))
        c1.Update()
        c1.SaveAs('{}/plot_xy.{}.pdf'.format(FIGURE_DIR, detectors[i]))

    input('Press any key to continue.')


def plot_pxpy(**kwargs):
    tf = TFile('LAPPD_MC_1g.1m.root')
    # tf = TFile('LAPPD_MC_1g.root')
    detectors = ['CollUpstream', 'Scraper1', 'Scraper2', 'D3', 'CollDownstream', 'NovaTarget']
    h_xys = []

    for detector in detectors:
        print('detector = {}'.format(detector))
        h_xy = TH2D('h_pxpy', 'h_pxpy', 80, -4, 4, 80, -4, 4)
        set_h2_style(h_xy)
        h_xys.append(h_xy)
        for particle in tf.Get('VirtualDetector/{}'.format(detector)):
            h_xy.Fill(particle.Px / 1000., particle.Py / 1000.)

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetRightMargin(0.18)
    gPad.SetLogz()
    set_h2_color_style()

    label_title_size = 32
    tex = TLatex()
    tex.SetTextFont(63)
    tex.SetTextSize(30)
    tex.SetTextAlign(33)
    tex.SetTextColor(kRed)

    for i, h_xy in enumerate(h_xys):
        h_xy.GetXaxis().SetTitle('P_{X} (GeV)')
        h_xy.GetYaxis().SetTitle('P_{Y} (GeV)')
        h_xy.GetXaxis().SetLabelSize(label_title_size)
        h_xy.GetYaxis().SetLabelSize(label_title_size)
        h_xy.GetZaxis().SetLabelSize(label_title_size)
        h_xy.GetXaxis().SetTitleSize(label_title_size)
        h_xy.GetYaxis().SetTitleSize(label_title_size)
        h_xy.GetZaxis().SetTitleSize(label_title_size)
        h_xy.Draw('colz')
        tex.DrawLatex(3.5, -3.1, 'Detector {}'.format(i + 1))

        c1.Update()
        c1.SaveAs('{}/plot_pxpy.{}.pdf'.format(FIGURE_DIR, detectors[i]))

    input('Press any key to continue.')


# 20180331_secondary_beam
gStyle.SetOptStat(0)
plot_pxpy()
# plot_xy()
# calculate_component_position()
# convert_rgb_color([201, 96, 35]) # quadrupole, color based on native values from mac color meter
# convert_rgb_color([9, 78, 149])  # dipole
# convert_rgb_color([78, 139, 50]) # trim
# convert_rgb_color([0, 204, 204]) # collimator
# convert_rgb_color([204, 0, 0])   # scraper
