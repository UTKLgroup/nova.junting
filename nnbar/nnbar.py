from rootalias import *
import math
import numpy as np
import random
import re
import os
from scipy.constants import Avogadro
from pprint import pprint
from math import sqrt
import json
import datetime


FIGURE_DIR = '/Users/juntinghuang/beamer/20190604_nnbar_data_quality/figures'
DATA_DIR = './data'

SECOND_IN_YEAR = 3.16           # 1.e7 s / year

COLORS = [
    kRed + 2,
    kGreen + 2,
    kBlue + 2,
    kYellow + 2,
    kMagenta + 2,
    kCyan + 2,
    kOrange + 2,
    kSpring + 2,
    kTeal + 2,
    kAzure + 2,
    kViolet + 2,
    kPink + 2
]

def get_gaussian(x, mu, sigma):
    return 1. / sigma / (2. * math.pi)**0.5 * math.exp(-0.5 * ((x - mu) / sigma)**2)


def get_poisson(x, k):
    return math.exp(-x) * x**k / math.factorial(k)


def get_xs(x_0, x_sigma):
    xs = []
    for i in [-2, -1, 0, 1, 2]:
        xs.append(x_0 + i * x_sigma)
    return xs


def get_super_kamiokande_parameters():
    parameters = {
        'event_count_observe': 24
    }

    center_values = {
        'exposure_0': 2.45,               # 1.e34 * neutron * year
        'efficiency_0': 12.1e-2,
        'background_0': 24.1,
        'suppression_factor_0': 0.517,    # 1.e23 s-1
    }

    errors = {
        'exposure_sigma': 3.e-2 * center_values['exposure_0'],
        'efficiency_sigma': 22.9e-2 * center_values['efficiency_0'],
        'background_sigma': 23.7e-2 * center_values['background_0'],
        'suppression_factor_sigma': 0.3 * center_values['suppression_factor_0'],
    }

    parameters.update(center_values)
    parameters.update(errors)
    # pprint(parameters)

    return parameters


def get_nova_parameters():
    parameters = {
        'event_count_observe': 50.
    }

    center_values = {
        'exposure_0': get_nova_exposure() / 1.e34, # 1.e34 * neutron * year
        'efficiency_0': 40.e-2,
        'background_0': 50.,
        'suppression_factor_0': 0.517 * 0.75,    # 1.e23 s-1
    }

    errors = {
        'exposure_sigma': 3.e-2 * center_values['exposure_0'],
        'efficiency_sigma': 22.9e-2 * center_values['efficiency_0'],
        'background_sigma': 23.7e-2 * center_values['background_0'],
        'suppression_factor_sigma': 0.3 * center_values['suppression_factor_0'],
    }

    parameters.update(center_values)
    parameters.update(errors)
    # pprint(parameters)

    return parameters


def get_life_time_bound(**kwargs):
    parameters = kwargs.get('parameters', get_super_kamiokande_parameters())
    make_plot = kwargs.get('make_plot', True)
    plot_suffix = kwargs.get('plot_suffix', '')
    max_x = kwargs.get('max_x', None)
    pprint(parameters)

    exposure_0 = parameters['exposure_0'] # 1.e34 * neutron * year
    efficiency_0 = parameters['efficiency_0']
    background_0 = parameters['background_0']
    suppression_factor_0 = parameters['suppression_factor_0']  # 1.e23 s-1
    exposure_sigma = parameters['exposure_sigma']
    efficiency_sigma = parameters['efficiency_sigma']
    background_sigma = parameters['background_sigma']
    suppression_factor_sigma = parameters['suppression_factor_sigma']
    event_count_observe = parameters['event_count_observe']

    exposures = get_xs(exposure_0, exposure_sigma)
    efficiencies = get_xs(efficiency_0, efficiency_sigma)
    backgrounds = get_xs(background_0, background_sigma)

    delta_event_rate_true = 0.1
    event_rate_trues = np.arange(0., 120., delta_event_rate_true)
    probabilities = []

    for event_rate_true in event_rate_trues:
        probability = 0.
        for exposure in exposures:
            for efficiency in efficiencies:
                for background in backgrounds:
                    event_count_true = event_rate_true * exposure * efficiency + background
                    probability += math.exp(-event_count_true) * event_count_true**event_count_observe \
                                   * get_gaussian(exposure, exposure_0, exposure_sigma) \
                                   * get_gaussian(efficiency, efficiency_0, efficiency_sigma) \
                                   * get_gaussian(background, background_0, background_sigma) \
                                   * exposure_sigma \
                                   * efficiency_sigma \
                                   * background_sigma
        probabilities.append(probability)

    total_area = 0.
    for i, event_rate_true in enumerate(event_rate_trues):
        total_area += probabilities[i] * delta_event_rate_true

    area = 0.
    confidence_level = 0.9
    event_rate_true_cl = None
    for i, event_rate_true in enumerate(event_rate_trues):
        area += probabilities[i] * delta_event_rate_true
        integral = area / total_area
        if integral > confidence_level:
            print('integral = ', integral)
            event_rate_true_cl = event_rate_true
            break

    life_time_cl = 1. / event_rate_true_cl * 1.e34
    print('event_rate_true_cl = {:.2e} per neutron per year at 90% C.L.'.format(event_rate_true_cl * 1.e-34))
    print('life_time_cl = {:.2e} neutron years at 90% C.L.'.format(life_time_cl))

    if make_plot:
        probabilities = list(map(lambda x: x / total_area, probabilities))
        gr = TGraph(len(event_rate_trues), event_rate_trues, np.array(probabilities))

        c1 = TCanvas('c1', 'c1', 800, 600)
        set_graph_style(gr)
        set_margin()

        gr.Draw('AL')
        gr.GetYaxis().SetDecimals()
        gr.GetXaxis().SetTitle('True Event Count per 10^{34} neutron year')
        gr.GetYaxis().SetTitle('Probability Density')
        gr.GetYaxis().SetTitleOffset(1.5)
        if not max_x:
            max_x = max(event_rate_trues)
        gr.GetXaxis().SetRangeUser(0, max_x)
        max_y = max(probabilities) * 1.1
        gr.GetYaxis().SetRangeUser(0, max_y)
        tl = TLine(event_rate_true_cl, 0, event_rate_true_cl, max_y)
        tl.SetLineWidth(3)
        tl.SetLineColor(kRed + 1)
        tl.SetLineStyle(7)
        tl.Draw()

        lg1 = TLegend(0.6, 0.8, 0.75, 0.88)
        set_legend_style(lg1)
        lg1.AddEntry(tl, '90% C.L. Limit', 'l')
        lg1.Draw()

        c1.Update()
        c1.SaveAs('{}/plot_life_time_bound{}.pdf'.format(FIGURE_DIR, plot_suffix))
        input('Press any key to continue.')

    return life_time_cl


def get_life_time_free(**kwargs):
    parameters = kwargs.get('parameters', get_super_kamiokande_parameters())
    make_plot = kwargs.get('make_plot', True)
    plot_suffix = kwargs.get('plot_suffix', '')
    max_x = kwargs.get('max_x', None)
    pprint(parameters)

    exposure_0 = parameters['exposure_0'] # 1.e34 * neutron * year
    efficiency_0 = parameters['efficiency_0']
    background_0 = parameters['background_0']
    suppression_factor_0 = parameters['suppression_factor_0'] # 1.e23 s-1
    exposure_sigma = parameters['exposure_sigma']
    efficiency_sigma = parameters['efficiency_sigma']
    background_sigma = parameters['background_sigma']
    suppression_factor_sigma = parameters['suppression_factor_sigma']
    event_count_observe = parameters['event_count_observe']

    exposures = get_xs(exposure_0, exposure_sigma)
    efficiencies = get_xs(efficiency_0, efficiency_sigma)
    backgrounds = get_xs(background_0, background_sigma)
    suppression_factors = get_xs(suppression_factor_0, suppression_factor_sigma)

    delta_inverse_square_free_life_time_true = 0.001
    inverse_square_free_life_time_trues = np.arange(0., 0.2, delta_inverse_square_free_life_time_true)
    probabilities = []

    for inverse_square_free_life_time_true in inverse_square_free_life_time_trues:
        probability = 0.
        for exposure in exposures:
            for efficiency in efficiencies:
                for background in backgrounds:
                    for suppression_factor in suppression_factors:
                        event_rate_true = inverse_square_free_life_time_true / suppression_factor * SECOND_IN_YEAR * 1.e2
                        event_count_true = event_rate_true * exposure * efficiency + background
                        probability += math.exp(-event_count_true) * event_count_true**event_count_observe \
                                       * get_gaussian(exposure, exposure_0, exposure_sigma) \
                                       * get_gaussian(efficiency, efficiency_0, efficiency_sigma) \
                                       * get_gaussian(background, background_0, background_sigma) \
                                       * get_gaussian(suppression_factor, suppression_factor_0, suppression_factor_sigma) \
                                       * exposure_sigma \
                                       * efficiency_sigma \
                                       * background_sigma \
                                       * suppression_factor_sigma
                                       # / math.factorial(event_count_observe)
        probabilities.append(probability)

    total_area = 0.
    for i, inverse_square_free_life_time_true in enumerate(inverse_square_free_life_time_trues):
        total_area += probabilities[i] * delta_inverse_square_free_life_time_true
    print('total_area = ', total_area)

    area = 0.
    confidence_level = 0.9
    inverse_square_free_life_time_true_cl = None
    for i, inverse_square_free_life_time_true in enumerate(inverse_square_free_life_time_trues):
        area += probabilities[i] * delta_inverse_square_free_life_time_true
        integral = area / total_area
        if integral > confidence_level:
            print('integral = ', integral)
            inverse_square_free_life_time_true_cl = inverse_square_free_life_time_true
            break

    free_life_time_cl = (1. / inverse_square_free_life_time_true_cl)**0.5 * 1.e8
    print('inverse_square_free_life_time_true_cl = {:.2e} s-2 at 90% C.L.'.format(inverse_square_free_life_time_true_cl * 1e-16))
    print('free_life_time_cl = {:.2e} seconds at 90% C.L.'.format(free_life_time_cl))

    if make_plot:
        probabilities = list(map(lambda x: x / total_area, probabilities))
        gr = TGraph(len(inverse_square_free_life_time_trues), inverse_square_free_life_time_trues, np.array(probabilities))

        c1 = TCanvas('c1', 'c1', 800, 600)
        set_graph_style(gr)
        set_margin()

        gr.Draw('AL')
        gr.GetYaxis().SetDecimals()
        gr.GetXaxis().SetTitle('1 / #tau^{2} (10^{-16} s^{-2})')
        gr.GetYaxis().SetTitle('Probability Density')
        gr.GetYaxis().SetTitleOffset(1.5)
        gr.GetYaxis().SetNdivisions(505, 1)
        gr.GetXaxis().SetNdivisions(505, 1)

        if not max_x:
            max_x = max(inverse_square_free_life_time_trues)
        gr.GetXaxis().SetRangeUser(0, max_x)
        max_y = max(probabilities) * 1.1
        gr.GetYaxis().SetRangeUser(0, max_y)
        tl = TLine(inverse_square_free_life_time_true_cl, 0, inverse_square_free_life_time_true_cl, max_y)
        tl.SetLineWidth(3)
        tl.SetLineColor(kRed + 1)
        tl.SetLineStyle(7)
        tl.Draw()

        lg1 = TLegend(0.6, 0.8, 0.75, 0.88)
        set_legend_style(lg1)
        lg1.AddEntry(tl, '90% C.L. Limit', 'l')
        lg1.Draw()

        c1.Update()
        c1.SaveAs('{}/plot_life_time_free{}.pdf'.format(FIGURE_DIR, plot_suffix))
        input('Press any key to continue.')

    return free_life_time_cl


def print_evd():
    events = {
        1: '\\bar{n} + p \\rightarrow \\pi^{+} + \\pi^{0}',
        2: '\\bar{n} + p \\rightarrow \\pi^{+} + 2\\pi^{0}',
        3: '\\bar{n} + p \\rightarrow \\pi^{+} + 3\\pi^{0}',
        4: '\\bar{n} + p \\rightarrow 2\\pi^{+} + \\pi^{-} + \\pi^{0}',
        5: '\\bar{n} + p \\rightarrow 2\\pi^{+} + \\pi^{-} + 2\\pi^{0}',
        6: '\\bar{n} + p \\rightarrow 2\\pi^{+} + \\pi^{-} + 2\\omega^{0}',
        7: '\\bar{n} + p \\rightarrow 3\\pi^{+} + 2\\pi^{-} + \\pi^{0}',
        8: '\\bar{n} + n \\rightarrow \\pi^{+} + \\pi^{-}',
        9: '\\bar{n} + n \\rightarrow 2\\pi^{0}',
        10: '\\bar{n} + n \\rightarrow \\pi^{+} + \\pi^{-} + \\pi^{0}',
        11: '\\bar{n} + n \\rightarrow \\pi^{+} + \\pi^{-} + 2\\pi^{0}',
        12: '\\bar{n} + n \\rightarrow \\pi^{+} + \\pi^{-} + 3\\pi^{0}',
        13: '\\bar{n} + n \\rightarrow 2\\pi^{+} + 2\\pi^{-}',
        14: '\\bar{n} + n \\rightarrow 2\\pi^{+} + 2\\pi^{-} + \\pi^{0}',
        15: '\\bar{n} + n \\rightarrow \\pi^{+} + \\pi^{-} + \\omega^{0}',
        16: '\\bar{n} + n \\rightarrow 2\\pi^{+} + 2\\pi^{-} + 2\\pi^{0}'
    }

    filenames = {
        1: 'nbarp_pi+_pi0',
        2: 'nbarp_pi+_2pi0',
        3: 'nbarp_pi+_3pi0',
        4: 'nbarp_2pi+_pi-_pi0',
        5: 'nbarp_2pi+_pi-_2pi0',
        6: 'nbarp_2pi+_pi-_2omega0',
        7: 'nbarp_3pi+_2pi-_pi0',
        8: 'nbarn_pi+_pi-',
        9: 'nbarn_2pi0',
        10: 'nbarn_pi+_pi-_pi0',
        11: 'nbarn_pi+_pi-_2pi0',
        12: 'nbarn_pi+_pi-_3pi0',
        13: 'nbarn_2pi+_2pi-',
        14: 'nbarn_2pi+_2pi-_pi0',
        15: 'nbarn_pi+_pi-_omega0',
        16: 'nbarn_2pi+_2pi-_2pi0'
    }

    for i in range(1, 17):
        with open('{}.txt'.format(filenames[i]), 'w') as f_caption:
            f_caption.write('A simulated event of ${}$ in the far detector.'.format(events[i]))

    with open('print_evd.tex', 'w') as f_evd:
        for i in range(1, 17):
            reaction = events[i]
            f_evd.write('\\begin{frame}\n')
            # f_evd.write('  \\frametitle{{Example of ${}$}}\n'.format(reaction))
            f_evd.write('  \\frametitle{{${}$}}\n'.format(reaction))
            f_evd.write('  \\begin{figure}\n')
            # f_evd.write('    \\includegraphics[scale = 0.25]{{figures/{{nnbar.{}}}.png}}\n'.format(i))
            # f_evd.write('    \\caption{{An example event of ${}$.}}\n'.format(reaction))
            f_evd.write('    \\includegraphics[scale = 0.25]{{figures/{{{}}}.png}}\n'.format(filenames[i]))
            f_evd.write('    \\caption{{A simulated event of ${}$.}}\n'.format(reaction))
            f_evd.write('  \\end{figure}\n')
            f_evd.write('\\end{frame}\n\n')
            f_evd.write('% .........................................................\n\n')


def plot(**kwargs):
    hist_name = kwargs.get('hist_name', 'fSliceCount')
    x_min = kwargs.get('x_min')
    x_max = kwargs.get('x_max')
    log_y = kwargs.get('log_y', False)
    log_x = kwargs.get('log_x', False)
    x_title = kwargs.get('x_title')
    y_title = kwargs.get('y_title')
    rebin = kwargs.get('rebin')
    normalize = kwargs.get('normalize', False)
    statbox_position = kwargs.get('statbox_position', 'right')
    root_filename = kwargs.get('root_filename')
    square_canvas = kwargs.get('square_canvas', False)

    tf = TFile('{}/{}'.format(data_dir, root_filename))
    h1 = tf.Get('nnbarana/{}'.format(hist_name))

    if rebin:
        h1.Rebin(rebin)

    if normalize:
        h1.Scale(1. / h1.Integral())

    canvas_height = 600
    if square_canvas:
        canvas_height = 800
    c1 = TCanvas('c1', 'c1', 800, canvas_height)
    set_margin()
    gPad.SetLogy(log_y)
    gPad.SetLogx(log_x)

    set_h1_style(h1)
    if x_min and x_max:
        h1.GetXaxis().SetRangeUser(x_min, x_max)
    if not log_y:
        h1.GetYaxis().SetRangeUser(0, get_max_y([h1]) * 1.1)
    if x_title:
        h1.GetXaxis().SetTitle(x_title)
    if y_title:
        h1.GetYaxis().SetTitle(y_title)
    h1.Draw('hist')

    c1.Update()
    draw_statbox(h1, x1=0.68)

    c1.Update()
    c1.SaveAs('{}/plot.{}.{}.pdf'.format(figure_dir, root_filename, hist_name))
    input('Press any key to continue.')


def plot_daq_hit(filename, **kwargs):
    draw_containment = kwargs.get('draw_containment', False)
    draw_option = kwargs.get('draw_option', 'box')

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_x = tf.Get('neutronoscana/fDaqHitXView')
    h_y = tf.Get('neutronoscana/fDaqHitYView')

    c1 = TCanvas('c1', 'c1', 1100, 800)
    set_h2_color_style()

    pad1 = TPad("pad1", "pad1", 0, 0.5, 1, 1)
    pad1.SetLeftMargin(0.1)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.2)
    pad1.SetBottomMargin(0.025)
    pad1.Draw()
    pad1.cd()

    set_h2_style(h_x)
    h_x.GetYaxis().SetTitle('X Cell Number')
    h_x.GetYaxis().SetTitleOffset(1.3)
    h_x.GetXaxis().SetLabelSize(0)
    h_x.GetXaxis().SetTitleSize(0)
    h_x.Draw(draw_option)
    if draw_option == 'colz':
        gPad.SetRightMargin(0.15)

    x_min = 4
    x_max = 377
    y_min = 6
    y_max = 347
    z_min = 3
    z_max = 891
    if draw_containment:
        lx_b = TLine(z_min, x_min, z_max, x_min)
        lx_t = TLine(z_min, x_max, z_max, x_max)
        lx_l = TLine(z_min, x_min, z_min, x_max)
        lx_r = TLine(z_max, x_min, z_max, x_max)
        lxs = [lx_b, lx_t, lx_l, lx_r]
        for lx in lxs:
            lx.SetLineWidth(1)
            lx.SetLineColor(kRed)
            lx.Draw('sames')

    c1.cd()
    pad2 = TPad('pad2', 'pad2', 0, 0, 1, 0.5)
    pad2.SetLeftMargin(0.1)
    pad2.SetRightMargin(0.05)
    pad2.SetBottomMargin(0.2)
    pad2.SetTopMargin(0.025)
    pad2.Draw()
    pad2.cd()

    set_h2_style(h_y)
    h_y.GetYaxis().SetTitle('Y Cell Number')
    h_y.GetYaxis().SetTitleOffset(1.3)
    h_y.GetXaxis().SetTitleOffset(2.2)
    h_y.Draw(draw_option)
    if draw_option == 'colz':
        gPad.SetRightMargin(0.15)

    if draw_containment:
        ly_b = TLine(z_min, y_min, z_max, y_min)
        ly_t = TLine(z_min, y_max, z_max, y_max)
        ly_l = TLine(z_min, y_min, z_min, y_max)
        ly_r = TLine(z_max, y_min, z_max, y_max)
        lys = [ly_b, ly_t, ly_l, ly_r]
        for ly in lys:
            ly.SetLineWidth(1)
            ly.SetLineColor(kRed)
            ly.Draw('sames')

    c1.Update()
    c1.SaveAs('{}/plot_daq_hit.{}.{}.pdf'.format(FIGURE_DIR, filename, draw_option))
    if draw_option == 'colz':
        c1.SaveAs('{}/plot_daq_hit.{}.{}.png'.format(FIGURE_DIR, filename, draw_option))
        c1.SaveAs('{}/plot_daq_hit.{}.{}.jpg'.format(FIGURE_DIR, filename, draw_option))
    input('Press any key to continue.')


def plot_daq_hit_colz(filename, **kwargs):
    draw_containment = kwargs.get('draw_containment', True)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_x = tf.Get('neutronoscana/fDaqHitXView')
    h_y = tf.Get('neutronoscana/fDaqHitYView')

    c1 = TCanvas('c1', 'c1', 1100, 800)
    set_h2_color_style()

    pad1 = TPad("pad1", "pad1", 0, 0.5, 1, 1)
    pad1.SetLeftMargin(0.1)
    pad1.SetRightMargin(0.15)
    pad1.SetTopMargin(0.2)
    pad1.SetBottomMargin(0.025)
    pad1.Draw()
    pad1.cd()
    gPad.SetLogz()

    set_h2_style(h_x)
    h_x.GetYaxis().SetTitle('X Cell Number')
    h_x.GetYaxis().SetTitleOffset(1.3)
    h_x.GetXaxis().SetLabelSize(0)
    h_x.GetXaxis().SetTitleSize(0)
    h_x.Draw('colz')

    x_min = 4
    x_max = 377
    y_min = 6
    y_max = 347
    z_min = 3
    z_max = 891
    if draw_containment:
        lx_b = TLine(z_min, x_min, z_max, x_min)
        lx_t = TLine(z_min, x_max, z_max, x_max)
        lx_l = TLine(z_min, x_min, z_min, x_max)
        lx_r = TLine(z_max, x_min, z_max, x_max)
        lxs = [lx_b, lx_t, lx_l, lx_r]
        for lx in lxs:
            lx.SetLineWidth(1)
            lx.SetLineColor(kRed)
            lx.Draw('sames')

    c1.cd()
    pad2 = TPad('pad2', 'pad2', 0, 0, 1, 0.5)
    pad2.SetLeftMargin(0.1)
    pad2.SetRightMargin(0.15)
    pad2.SetBottomMargin(0.2)
    pad2.SetTopMargin(0.025)
    pad2.Draw()
    pad2.cd()
    gPad.SetLogz()

    set_h2_style(h_y)
    h_y.GetYaxis().SetTitle('Y Cell Number')
    h_y.GetYaxis().SetTitleOffset(1.3)
    h_y.GetXaxis().SetTitleOffset(2.2)
    h_y.Draw('colz')

    if draw_containment:
        ly_b = TLine(z_min, y_min, z_max, y_min)
        ly_t = TLine(z_min, y_max, z_max, y_max)
        ly_l = TLine(z_min, y_min, z_min, y_max)
        ly_r = TLine(z_max, y_min, z_max, y_max)
        lys = [ly_b, ly_t, ly_l, ly_r]
        for ly in lys:
            ly.SetLineWidth(1)
            ly.SetLineColor(kRed)
            ly.Draw('sames')

    c1.Update()
    # c1.SaveAs('{}/plot_daq_hit.{}.pdf'.format(FIGURE_DIR, filename))
    c1.SaveAs('{}/plot_daq_hit.{}.png'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def get_track3d(tf):
    gr_xs = []
    gr_ys = []
    track_count = 0
    for track in tf.Get('neutronoscana/fTrack3dTree'):
        track_count += 1
        # if (track.fTrack3dStartY == 0. and track.fTrack3dEndY == 0.) or (track.fTrack3dStartX == 0. and track.fTrack3dEndX == 0.):
            # continue
        gr_x = TGraph(2, np.array([track.fTrack3dStartZ, track.fTrack3dEndZ]), np.array([track.fTrack3dStartX, track.fTrack3dEndX]))
        gr_y = TGraph(2, np.array([track.fTrack3dStartZ, track.fTrack3dEndZ]), np.array([track.fTrack3dStartY, track.fTrack3dEndY]))

        color = random.choice(COLORS)
        gr_x.SetLineWidth(2)
        gr_y.SetLineWidth(2)
        gr_x.SetLineColor(color)
        gr_y.SetLineColor(color)

        gr_xs.append(gr_x)
        gr_ys.append(gr_y)

    print('track_count = {}'.format(track_count))
    return gr_xs, gr_ys


def get_track2d(tf):
    gr_xs = []
    gr_ys = []
    track_count = 0
    for track in tf.Get('neutronoscana/fTrack2dTree'):
        track_count += 1

        color = random.choice(COLORS)
        if track.fTrack2dView == 1:
            gr_x = TGraph(2, np.array([track.fTrack2dStartZ, track.fTrack2dEndZ]), np.array([track.fTrack2dStartV, track.fTrack2dEndV]))
            gr_x.SetLineWidth(2)
            gr_x.SetLineColor(color)
            gr_xs.append(gr_x)
        else:
            gr_y = TGraph(2, np.array([track.fTrack2dStartZ, track.fTrack2dEndZ]), np.array([track.fTrack2dStartV, track.fTrack2dEndV]))
            gr_y.SetLineWidth(2)
            gr_y.SetLineColor(color)
            gr_ys.append(gr_y)

    print('track_count = {}'.format(track_count))
    return gr_xs, gr_ys


def plot_track(filename, **kwargs):
    dimension = kwargs.get('dimension', '2d')
    draw_track = kwargs.get('draw_track', True)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_x = tf.Get('neutronoscana/fDaqHitXView')
    h_y = tf.Get('neutronoscana/fDaqHitYView')
    gr_xs = None
    gr_ys = None
    if dimension == '2d':
        gr_xs, gr_ys = get_track2d(tf)
    elif dimension == '3d':
        gr_xs, gr_ys = get_track3d(tf)

    c1 = TCanvas('c1', 'c1', 1100, 800)
    set_h2_color_style()

    pad1 = TPad("pad1", "pad1", 0, 0.5, 1, 1)
    pad1.SetLeftMargin(0.1)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.2)
    pad1.SetBottomMargin(0.025)
    pad1.Draw()
    pad1.cd()

    set_h2_style(h_x)
    h_x.GetYaxis().SetTitle('X Cell Number')
    h_x.GetYaxis().SetTitleOffset(1.3)
    h_x.GetXaxis().SetLabelSize(0)
    h_x.GetXaxis().SetTitleSize(0)
    h_x.Draw('box')
    if draw_track:
        for gr_x in gr_xs:
            gr_x.Draw('L, sames')

    c1.cd()
    pad2 = TPad('pad2', 'pad2', 0, 0, 1, 0.5)
    pad2.SetLeftMargin(0.1)
    pad2.SetRightMargin(0.05)
    pad2.SetBottomMargin(0.2)
    pad2.SetTopMargin(0.025)
    pad2.Draw()
    pad2.cd()

    set_h2_style(h_y)
    h_y.GetYaxis().SetTitle('Y Cell Number')
    h_y.GetYaxis().SetTitleOffset(1.3)
    h_y.GetXaxis().SetTitleOffset(2.2)
    h_y.Draw('box')
    if draw_track:
        for gr_y in gr_ys:
            gr_y.Draw('L, sames')

    c1.Update()
    c1.SaveAs('{}/plot_track.{}.draw_track_{}.pdf'.format(FIGURE_DIR, filename, draw_track))
    input('Press any key to continue.')


def plot_track_theta_variance():
    tf_cosmic = TFile('{}/neutronosc_ddt_hist.cosmic.root'.format(DATA_DIR))
    tf_clean = TFile('{}/neutronosc_ddt_hist.clean.root'.format(DATA_DIR))

    h_cosmic = tf_cosmic.Get('neutronoscana/fTrackThetaVarianceXY')
    h_clean = tf_clean.Get('neutronoscana/fTrackThetaVarianceXY')

    print('h_cosmic.Integral() = {}'.format(h_cosmic.Integral(1, 1001, 1, 1001)))
    print('h_clean.Integral() = {}'.format(h_clean.Integral(1, 1001, 1, 1001)))
    h_clean.Scale(1. / h_clean.Integral(1, 1001, 1, 1001))
    h_cosmic.Scale(1. / h_cosmic.Integral(1, 1001, 1, 1001))

    print('1 - h_clean.Integral(1, 10, 1, 10) = {}'.format(1 - h_clean.Integral(1, 10, 1, 10)))
    print('1 - h_cosmic.Integral(1, 10, 1, 10) = {}'.format(1 - h_cosmic.Integral(1, 10, 1, 10)))

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetRightMargin(0.2)
    gPad.SetLogx()
    gPad.SetLogy()
    gPad.SetLogz()
    set_h2_color_style()
    set_h2_style(h_cosmic)
    h_cosmic.Draw('colz')
    h_cosmic.GetXaxis().SetTitleOffset(1.4)
    h_cosmic.GetYaxis().SetTitleOffset(1.6)
    c1.Update()
    c1.SaveAs('{}/plot_track_theta_variance.cosmic.pdf'.format(FIGURE_DIR))

    c2 = TCanvas('c2', 'c2', 600, 600)
    set_margin()
    gPad.SetRightMargin(0.2)
    gPad.SetLogx()
    gPad.SetLogy()
    gPad.SetLogz()
    set_h2_color_style()
    set_h2_style(h_clean)
    h_clean.Draw('colz')
    h_clean.GetXaxis().SetTitleOffset(1.4)
    h_clean.GetYaxis().SetTitleOffset(1.6)

    c2.Update()
    c2.SaveAs('{}/plot_track_theta_variance.clean.pdf'.format(FIGURE_DIR))

    input('Press any key to continue.')


def plot_slice_track_count():
    tf_cosmic = TFile('{}/neutronosc_ddt_hist.cosmic.root'.format(DATA_DIR))
    tf_clean = TFile('{}/neutronosc_ddt_hist.clean.root'.format(DATA_DIR))

    h_cosmic = tf_cosmic.Get('neutronoscana/fSliceTrackCountXY')
    h_clean = tf_clean.Get('neutronoscana/fSliceTrackCountXY')
    h_clean.Scale(1. / h_clean.Integral())
    h_cosmic.Scale(1. / h_cosmic.Integral())

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetRightMargin(0.2)
    set_h2_color_style()
    set_h2_style(h_cosmic)
    h_cosmic.GetXaxis().SetRangeUser(0, 30)
    h_cosmic.GetYaxis().SetRangeUser(0, 30)
    h_cosmic.Draw('colz')
    c1.Update()
    c1.SaveAs('{}/plot_slice_track_count.cosmic.pdf'.format(FIGURE_DIR))

    c2 = TCanvas('c2', 'c2', 600, 600)
    set_margin()
    gPad.SetRightMargin(0.2)
    set_h2_color_style()
    set_h2_style(h_clean)
    h_clean.GetXaxis().SetRangeUser(0, 30)
    h_clean.GetYaxis().SetRangeUser(0, 30)
    h_clean.Draw('colz')

    c2.Update()
    c2.SaveAs('{}/plot_slice_track_count.clean.pdf'.format(FIGURE_DIR))

    input('Press any key to continue.')


def plot_1d_cut(hist_name, **kwargs):
    cosmic_filename = kwargs.get('cosmic_filename', 'neutronosc_ddt_hist.cosmic.root')
    signal_filename = kwargs.get('signal_filename', 'neutronosc_ddt_hist.clean.root')
    x_max = kwargs.get('x_max', None)
    y_max = kwargs.get('y_max', None)
    rebin = kwargs.get('rebin', None)
    x_cut = kwargs.get('x_cut', None)
    log_x = kwargs.get('log_x', False)
    log_y = kwargs.get('log_y', False)
    legend_fx1ndc = kwargs.get('legend_fx1ndc', 0.57)
    legend_fx2ndc = kwargs.get('legend_fx2ndc', 0.84)
    legend_left = kwargs.get('legend_left', False)
    legend_text_cosmic = kwargs.get('legend_text_cosmic', 'Cosmic ray')
    legend_text_signal = kwargs.get('legend_text_signal', 'Signal')
    tdirectory_cosmic = kwargs.get('tdirectory_cosmic', 'neutronoscana')
    tdirectory_signal = kwargs.get('tdirectory_signal', 'neutronoscana')

    tf_cosmic = TFile('{}/{}'.format(DATA_DIR, cosmic_filename))
    tf_clean = TFile('{}/{}'.format(DATA_DIR, signal_filename))

    h_cosmic = tf_cosmic.Get('{}/{}'.format(tdirectory_cosmic, hist_name))
    h_clean = tf_clean.Get('{}/{}'.format(tdirectory_signal, hist_name))

    print('h_clean.GetNbinsX() = {}'.format(h_clean.GetNbinsX()))
    print('h_clean.Integral(1, h_clean.FindBin(x_cut)) / h_clean.Integral() = {}'.format(h_clean.Integral(1, h_clean.FindBin(x_cut)) / h_clean.Integral()))
    print('h_cosmic.Integral(1, h_cosmic.FindBin(x_cut)) / h_cosmic.Integral() = {}'.format(h_cosmic.Integral(1, h_cosmic.FindBin(x_cut)) / h_cosmic.Integral()))
    print('1. - h_clean.Integral(1, h_clean.FindBin(x_cut)) / h_clean.Integral() = {}'.format(1. - h_clean.Integral(1, h_clean.FindBin(x_cut)) / h_clean.Integral()))
    print('1. - h_cosmic.Integral(1, h_cosmic.FindBin(x_cut)) / h_cosmic.Integral() = {}'.format(1. - h_cosmic.Integral(1, h_cosmic.FindBin(x_cut)) / h_cosmic.Integral()))

    if rebin:
        h_cosmic.Rebin(rebin)
        h_clean.Rebin(rebin)

    h_clean.Scale(1. / h_clean.Integral())
    h_cosmic.Scale(1. / h_cosmic.Integral())

    c1 = TCanvas('c1', 'c1', 800, 600)
    c1.SetTitle(hist_name)

    set_margin()
    if log_x:
        gPad.SetLogx()
    if log_y:
        gPad.SetLogy()

    set_h1_style(h_cosmic)
    h_cosmic.Draw('hist')
    if x_max:
        h_cosmic.GetXaxis().SetRangeUser(0, x_max)

    if not y_max:
        y_max = max(h_cosmic.GetMaximum(), h_clean.GetMaximum()) * 1.2
    h_cosmic.GetYaxis().SetRangeUser(0, y_max)

    set_h1_style(h_clean)
    h_clean.SetLineColor(kRed + 2)
    h_clean.Draw('hist,sames')

    lg1 = None
    if legend_left:
        lg1 = TLegend(0.21, 0.7, 0.48, 0.85)
    else:
        lg1 = TLegend(legend_fx1ndc, 0.7, legend_fx2ndc, 0.85)
    set_legend_style(lg1)
    lg1.AddEntry(h_cosmic, legend_text_cosmic, 'l')
    lg1.AddEntry(h_clean, legend_text_signal, 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_1d_cut.{}.{}.pdf'.format(FIGURE_DIR, cosmic_filename, hist_name))
    input('Press any key to continue.')


def get_integral_under_curve(tf1, h1):
    integral = 0.
    for i in range(1, h1.GetNbinsX() + 1):
        for j in range(1, h1.GetNbinsY() + 1):
            x = h1.GetXaxis().GetBinCenter(i)
            y = h1.GetYaxis().GetBinCenter(j)
            if y < tf1.Eval(x):
                integral += h1.GetBinContent(i, j)
    return integral


def plot_2d_cuts(histname, **kwargs):
    cosmic_filename = kwargs.get('cosmic_filename', 'neutronosc_ddt_hist.cosmic.root')
    signal_filename = kwargs.get('signal_filename', 'neutronosc_ddt_hist.clean.root')
    log_x = kwargs.get('log_x', True)
    log_y = kwargs.get('log_y', True)
    log_z = kwargs.get('log_z', True)
    x_max = kwargs.get('x_max', None)
    y_max = kwargs.get('y_max', None)
    x_cut = kwargs.get('x_cut', 0.)
    y_cut = kwargs.get('y_cut', 0.)
    cosmic_only = kwargs.get('cosmic_only', False)
    signal_only = kwargs.get('signal_only', False)
    rebin = kwargs.get('rebin', None)
    grid = kwargs.get('grid', False)
    equation = kwargs.get('equation', None)

    tf_cosmic = TFile('{}/{}'.format(DATA_DIR, cosmic_filename))
    tf_clean = TFile('{}/{}'.format(DATA_DIR, signal_filename))

    h_cosmic = tf_cosmic.Get('neutronoscana/{}'.format(histname))
    h_clean = tf_clean.Get('neutronoscana/{}'.format(histname))

    if x_cut > 0. and y_cut > 0.:
        x_cut_bin = h_cosmic.GetXaxis().FindBin(x_cut)
        y_cut_bin = h_cosmic.GetYaxis().FindBin(y_cut)
        print('x_cut_bin = {}'.format(x_cut_bin))
        print('y_cut_bin = {}'.format(y_cut_bin))
        print('h_cosmic.Integral() = {}'.format(h_cosmic.Integral()))
        print('h_clean.Integral() = {}'.format(h_clean.Integral()))
        print('1 - h_cosmic.Integral(1, x_cut_bin, 1, y_cut_bin) / h_cosmic.Integral() = {}'.format(1 - h_cosmic.Integral(1, x_cut_bin, 1, y_cut_bin) / h_cosmic.Integral()))
        print('1 - h_clean.Integral(1, x_cut_bin, 1, y_cut_bin) / h_clean.Integral() = {}'.format(1 - h_clean.Integral(1, x_cut_bin, 1, y_cut_bin) / h_clean.Integral()))

    if rebin:
        h_cosmic.Rebin2D(rebin, rebin)
        h_clean.Rebin2D(rebin, rebin)

    h_clean.Scale(1. / h_clean.Integral())
    h_cosmic.Scale(1. / h_cosmic.Integral())

    # for width-length ratio
    # equation = '-3.125 * pow(x - 0.4, 3)'

    # for multiple cell fraction
    # equation = '0.1 * exp(-x / 0.05) + 0.02'

    # for plane cell asymmetry
    # equation = '-0.6*x + 0.6'

    tf1 = None
    if equation:
        tf1 = TF1('tf1', equation, 0, 1)
        tf1.SetLineWidth(3)
        tf1.SetLineColor(kRed)

    if tf1:
        print('integral under equation:')
        print('signal: {}'.format(get_integral_under_curve(tf1, h_clean)))
        print('cosmic: {}'.format(get_integral_under_curve(tf1, h_cosmic)))

    if not signal_only:
        c1 = TCanvas('c1', 'c1', 600, 600)
        set_margin()
        gPad.SetRightMargin(0.2)
        if grid:
            gPad.SetGrid()
        if log_x:
            gPad.SetLogx()
        if log_y:
            gPad.SetLogy()
        if log_z:
            gPad.SetLogz()
        set_h2_color_style()
        set_h2_style(h_cosmic)
        h_cosmic.Draw('colz')
        h_cosmic.GetXaxis().SetTitleOffset(1.4)
        h_cosmic.GetYaxis().SetTitleOffset(1.6)
        if x_max:
            h_cosmic.GetXaxis().SetRangeUser(0., x_max)
        if y_max:
            h_cosmic.GetYaxis().SetRangeUser(0., y_max)
        if tf1:
            tf1.Draw('sames')
        c1.Update()
        c1.SaveAs('{}/plot_2d_cuts.{}.{}.cosmic.pdf'.format(FIGURE_DIR, cosmic_filename, histname))

    if not cosmic_only:
        c2 = TCanvas('c2', 'c2', 600, 600)
        set_margin()
        gPad.SetRightMargin(0.2)
        if grid:
            gPad.SetGrid()
        if log_x:
            gPad.SetLogx()
        if log_y:
            gPad.SetLogy()
        if log_z:
            gPad.SetLogz()
        set_h2_color_style()
        set_h2_style(h_clean)
        h_clean.Draw('colz')
        h_clean.GetXaxis().SetTitleOffset(1.4)
        h_clean.GetYaxis().SetTitleOffset(1.6)
        if x_max:
            h_clean.GetXaxis().SetRangeUser(0., x_max)
        if y_max:
            h_clean.GetYaxis().SetRangeUser(0., y_max)
        if tf1:
            tf1.Draw('sames')
        c2.Update()
        c2.SaveAs('{}/plot_2d_cuts.{}.{}.clean.pdf'.format(FIGURE_DIR, signal_filename, histname))

    input('Press any key to continue.')


def plot_daq_hit_1d(filename, hist_name):
    f1 = TFile('{}/{}'.format(DATA_DIR, filename))
    h1 = f1.Get('neutronoscana/{}'.format(hist_name))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogz()
    gPad.SetRightMargin(0.2)

    set_h2_color_style()
    set_h2_style(h1)
    h1.GetXaxis().SetRangeUser(300, 600)
    h1.GetXaxis().SetTitleOffset(1.2)
    h1.GetYaxis().SetTitleOffset(1.2)
    h1.GetZaxis().SetTitle('ADC')
    h1.Draw('colz')

    c1.Update()
    c1.SaveAs('{}/plot_daq_hit_1d.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def calculate_trigger_rate():
    event_count = 5031.
    event_duration = 0.55e-3    # s
    exposure = event_count * event_duration

    cut_names = ['pre-containment', 'containment', 'width-length ratio', 'cell number multiplicity', 'hit count asymmetry', 'FEB flasher', 'one-planer']
    slice_counts = [
        328862,
        8840,
        121,
        81,
        56,
        18,
        13
    ]

    fractions = [slice_count / slice_counts[0] for slice_count in slice_counts]
    rates = [slice_count / exposure for slice_count in slice_counts]
    for i, fraction in enumerate(fractions):
        rate = '{:.0f}'.format(rates[i])
        print('{} & {:.0f} & \SI{{{}}}{{}} & \SI{{{}}}{{}} \\\\'.format(cut_names[i], slice_counts[i], '{:.1E}'.format(fractions[i]), rate))


def calculate_efficiency():
    cut_names = ['pre-containment', 'containment', 'width-length ratio', 'cell number multiplicity', 'hit count asymmetry', 'FEB flasher', 'one-planer']
    slice_counts = [
	10104,
	6814,
        6117,
        5970,
        5955,
        5901,
        5898
    ]
    fractions = [slice_count / slice_counts[0] for slice_count in slice_counts]
    for i, fraction in enumerate(fractions):
        fraction = fractions[i] * 100.
        print('{} & {:.0f} & {:.0f}\\% \\\\'.format(cut_names[i], slice_counts[i], fraction))


def test_integral():
    # h1.Integral() does not include overflow and underflow bins
    # h1.Integral() is affected by h1.GetXaxis().SetRangeUser()
    # bin 0 is the underflow bin, bin (last bin + 1) is overflow bin
    # the value on bin edge goes up, e.g. 0 goes to the bin of 0 to 1

    h1 = TH1D('h1', 'h1', 3, 0, 3)
    h1.Fill(1)
    h1.Fill(-1)
    h1.Fill(4)
    print('h1.Integral() = {}'.format(h1.Integral()))
    print('h1.Integral(1, 3) = {}'.format(h1.Integral(1, 3)))
    print('h1.Integral(1, 4) = {}'.format(h1.Integral(1, 4)))
    print('h1.Integral(0, 4) = {}'.format(h1.Integral(0, 4)))
    print('h1.Integral(-1, 4)= {}'.format(h1.Integral(-1, 4)))
    print('h1.Integral(-1, 5)= {}'.format(h1.Integral(-1, 5)))

    h1.Fill(0)
    print('h1.Fill(0)')
    print('h1.Integral() = {}'.format(h1.Integral()))
    print('h1.Integral(1, 3) = {}'.format(h1.Integral(1, 3)))
    h1.GetXaxis().SetRangeUser(0, 1)
    print('h1.GetXaxis().SetRangeUser(0, 1)')
    print('h1.Integral() = {}'.format(h1.Integral()))
    print('h1.Integral(1, 3) = {}'.format(h1.Integral(1, 3)))


def check_dungs_topology_cut():
    countXWithSingleHit = 0
    # cellNumbers = [10, 5, 10, 5, 5, 6, 10, 10]
    # cellNumbers = [10, 5, 10, 5, 5, 6, 10]
    cellNumbers = [10, 5, 10, 5, 5, 6, 10, 10, 10]
    len_cellNumbers = len(cellNumbers)

    for j in range(len_cellNumbers):
        test = 0
        for k in range(j + 1, len_cellNumbers):
            if cellNumbers[k] == cellNumbers[j]:
                k = len_cellNumbers
            test = k
            if k == len_cellNumbers:
                break
        if test == len_cellNumbers:
            countXWithSingleHit += 1

    print('countXWithSingleHit = {}'.format(countXWithSingleHit))
    print('len_cellNumbers = {}'.format(len_cellNumbers))
    print('countXWithSingleHit / len_cellNumbers = {}'.format(countXWithSingleHit / len_cellNumbers))


def plot_containment_effect():
    event_count = 1636.
    event_duration = 0.55e-3    # s
    exposure = event_count * event_duration

    vetos = []
    cosmic_ray_rates = []
    signal_efficiencies = []
    cosmic_ray_cut_rates = []
    signal_cut_efficiencies = []

    ymaxs = [375., 371., 367., 363., 359., 355., 351., 347., 343.]
    for ymax in ymaxs:
        cosmic_total, cosmic_containment, cosmic_cut = get_event_count('y_max_{}_cosmic.log'.format(int(ymax)))
        clean_total, clean_containment, clean_cut = get_event_count('y_max_{}_clean.log'.format(int(ymax)))
        cosmic_ray_rates.append(cosmic_containment / exposure)
        signal_efficiencies.append(clean_containment / clean_total)
        cosmic_ray_cut_rates.append(cosmic_cut / exposure)
        signal_cut_efficiencies.append(clean_cut / clean_total)
        vetos.append(384 - ymax)

    gr_cosmic_rate = TGraph(len(vetos), np.array(vetos), np.array(cosmic_ray_rates))
    gr_signal_efficiency = TGraph(len(vetos), np.array(vetos), np.array(signal_efficiencies))
    gr_cosmic_cut_rate = TGraph(len(vetos), np.array(vetos), np.array(cosmic_ray_cut_rates))
    gr_signal_cut_efficiency = TGraph(len(vetos), np.array(vetos), np.array(signal_cut_efficiencies))

    grs = [
        gr_cosmic_rate,
        gr_cosmic_cut_rate,
        gr_signal_efficiency,
        gr_signal_cut_efficiency
    ]

    gr_names = [
        'gr_cosmic_rate',
        'gr_cosmic_cut_rate',
        'gr_signal_efficiency',
        'gr_signal_cut_efficiency'
    ]

    for i, gr in enumerate(grs):
        set_graph_style(gr)
        gr.GetXaxis().SetTitle('Top Containment Cell Count')
        gr.GetYaxis().SetTitleOffset(2.)
        if i < 2:
            gr.GetYaxis().SetTitle('Trigger Rate (Hz)')
        else:
            gr.GetYaxis().SetTitle('Signal Efficiency')

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetGrid()
    gPad.SetLeftMargin(0.2)

    # gr_cosmic_rate.Draw('AL')
    # c1.Update()

    for i, gr in enumerate(grs):
        gr.Draw('AL')
        c1.Update()
        c1.SaveAs('{}/plot_containment_effect.{}.pdf'.format(FIGURE_DIR, gr_names[i]))

    input('Press any key to continue.')


def get_event_count(filename):
    with open('{}/{}'.format(DATA_DIR, filename)) as f_log:
        number_of_slices = []
        remain_slices = []
        for line in f_log:
            if re.match(".*Number of remained slices:.*", line):
                remain_slices.append(int(line.split(':')[1].strip()))
            if re.match(".*Number of slices:.*", line):
                number_of_slices.append(int(line.split(':')[1].strip()))

        # print('number_of_slices = {}'.format(number_of_slices))
        # print('remain_slices = {}'.format(remain_slices))
        return number_of_slices[0], number_of_slices[1], remain_slices[0]

def print_trigger_evd_tex():
    figure_names = [
        'run_25148.subRun_24.event_9032.2018_07_20.16_22_19.png',
        'run_25148.subRun_24.event_3936.2018_07_20.16_18_44.png',
        'run_25148.subRun_24.event_3427.2018_07_20.16_10_34.png',
        'run_25148.subRun_24.event_2684.2018_07_20.16_07_16.png',
        'run_25148.subRun_24.event_2295.2018_07_20.15_53_21.png',
        'run_24679.subRun_8.event_6290.2018_07_20.15_45_48.png',
        'run_24679.subRun_8.event_4968.2018_07_20.15_40_41.png',
        'run_24679.subRun_8.event_227.2018_07_20.15_34_00.png',
        'run_24679.subRun_8.event_524.2018_07_20.15_29_10.png',
        'run_24679.subRun_8.event_1389.2018_07_20.15_26_35.png',
        'run_24679.subRun_8.event_4739.2018_07_20.15_19_40.png',
        'run_25154.subRun_26.event_2129.2018_07_21.22_00_26.png',
        'run_25154.subRun_26.event_1900.2018_07_21.21_48_58.png'
    ]

    run_subrun_events = [
        'run_24679.subRun_8.event_227',
        'run_24679.subRun_8.event_524',
        'run_24679.subRun_8.event_1389',
        'run_24679.subRun_8.event_4739',
        'run_24679.subRun_8.event_4968',
        'run_24679.subRun_8.event_6290',
        'run_25148.subRun_24.event_2295',
        'run_25148.subRun_24.event_2684',
        'run_25148.subRun_24.event_3427',
        'run_25148.subRun_24.event_3936',
        'run_25148.subRun_24.event_9032',
        'run_25154.subRun_26.event_1900',
        'run_25154.subRun_26.event_2129'
    ]

    with open('/Users/juntinghuang/beamer/20180719_nnbar_globalconfig/evd.tex', 'w') as f_tex:
        for i, run_subrun_event in enumerate(run_subrun_events):
            info = run_subrun_event.split('.')
            run = info[0].split('_')[1]
            subrun = info[1].split('_')[1]
            event = info[2].split('_')[1]

            figure = None
            for figure_name in figure_names:
                if run_subrun_event in figure_name:
                    figure = figure_name

            if not figure:
                continue

            f_tex.write('\n% .........................................................\n\n')
            f_tex.write('\\begin{frame}\n')
            f_tex.write('  \\frametitle{{Run {}, Subrun {}, Event {}}}\n'.format(run, subrun, event))
            f_tex.write('  \\vspace{6mm}\n')
            f_tex.write('  \\begin{figure}\n')
            f_tex.write('    \\includegraphics[width = \\textwidth]{{figures/evd/{{{}}}.png}}\n'.format(os.path.splitext(figure)[0]))
            f_tex.write('    \\caption{{A triggered slice in Run {}, Subrun {}, Event {}.}}\n'.format(run, subrun, event))
            f_tex.write('  \\end{figure}\n')
            f_tex.write('\\end{frame}\n')


def plot_feb_flasher(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_x = tf.Get('neutronoscana/fDaqHitXView')
    h_y = tf.Get('neutronoscana/fDaqHitYView')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetRightMargin(0.2)

    set_h2_color_style()
    set_h2_style(h_x)
    h_x.GetZaxis().SetTitle('Raw ADC')
    h_x.GetYaxis().SetTitle('X Cell Number')
    h_x.GetYaxis().SetTitleOffset(1.3)
    h_x.Draw('colz')
    h_x.GetXaxis().SetRangeUser(790, 820)
    h_x.GetYaxis().SetRangeUser(280, 330)

    c1.Update()
    c1.SaveAs('{}/plot_feb_flasher.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')

def plot_one_planer(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_x = tf.Get('neutronoscana/fDaqHitXView')
    h_y = tf.Get('neutronoscana/fDaqHitYView')

    c1 = TCanvas('c1', 'c1', 1000, 600)
    set_margin()
    # gPad.SetRightMargin(0.2)

    set_h2_color_style()
    set_h2_style(h_y)
    h_y.GetZaxis().SetTitle('Raw ADC')
    h_y.GetYaxis().SetTitle('Y Cell Number')
    h_y.GetYaxis().SetTitleOffset(1.3)
    h_y.Draw('box')
    # h_y.GetXaxis().SetRangeUser(790, 820)
    # h_y.GetYaxis().SetRangeUser(280, 330)

    c1.Update()
    c1.SaveAs('{}/plot_one_planer.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_nova_exposure():
    detector_mass = 1.4e10      # g
    mineral_oil_mass_fraction = 0.65
    pvc_mass_fraction = 0.35
    neutron_count_in_carbon = 6
    carbon_molar_mass = 12.     # g / mol
    neutron_count = detector_mass * (mineral_oil_mass_fraction * 12. / 14. + pvc_mass_fraction * 24. / (24. + 3. + 35.)) / carbon_molar_mass * Avogadro * 6.
    year_count = 6.
    neutron_year = neutron_count * year_count

    print('neutron_count = {}'.format(neutron_count))
    print('neutron_year = {}'.format(neutron_year))

    return neutron_year


def plot_nova_sensitivity():
    parameters = get_nova_parameters()
    # parameters = get_super_kamiokande_parameters()
    parameters['event_count_observe'] = 50.
    parameters['background_0'] = 50.
    parameters['efficiency_0'] = 0.4
    life_times = []

    efficiency_0s = []
    for efficiency_0 in np.arange(0.1, 0.7, 0.1):
        print('\n*******************')
        parameters['efficiency_0'] = efficiency_0
        life_time = get_life_time_free(parameters=parameters, make_plot=False)
        # life_time = get_life_time_bound(parameters=parameters, make_plot=False)
        print('life_time = {}'.format(life_time))
        life_times.append(life_time)
        efficiency_0s.append(efficiency_0 * 100.)
    gr = TGraph(len(efficiency_0s), np.array(efficiency_0s), np.array(life_times))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_graph_style(gr)

    gr.GetXaxis().SetRangeUser(efficiency_0s[0], efficiency_0s[-1])
    gr.GetXaxis().SetTitle('NOvA Signal Efficiency (%)')
    gr.GetYaxis().SetTitle('#tau (s)')
    gr.Draw('AL')

    life_time_super_kamiokande = 2.7e8
    tl = TLine(efficiency_0s[0], life_time_super_kamiokande, efficiency_0s[-1], life_time_super_kamiokande)
    tl.SetLineWidth(3)
    tl.SetLineColor(kRed + 1)
    tl.SetLineStyle(7)
    tl.Draw()

    lg1 = TLegend(0.19, 0.7, 0.34, 0.85)
    set_legend_style(lg1)
    lg1.AddEntry(gr, 'NOvA limit', 'l')
    lg1.AddEntry(tl, 'Super-Kamiokande limit', 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_nova_sensitivity.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_lifetime_vs_background():
    exposure_year = get_nova_exposure()
    exposure_second = exposure_year * SECOND_IN_YEAR * 1.e7
    print('exposure_year = {}'.format(exposure_year))
    print('exposure_second = {}'.format(exposure_second))

    efficiency = 0.3
    suppression_factor = 0.388e23

    backgrounds = np.arange(10., 1000., 10.)
    bound_lifetimes = []
    free_lifetimes = []
    for background in backgrounds:
        bound_lifetime = exposure_year * 0.3 / sqrt(background) # year
        free_lifetime = sqrt(bound_lifetime * SECOND_IN_YEAR * 1.e7 / suppression_factor)

        bound_lifetimes.append(bound_lifetime)
        free_lifetimes.append(free_lifetime)

    gr_bound = TGraph(len(backgrounds), np.array(backgrounds), np.array(bound_lifetimes))
    gr_free = TGraph(len(backgrounds), np.array(backgrounds), np.array(free_lifetimes))

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetLogx()
    gPad.SetGrid()
    set_graph_style(gr_free)

    gr_free.GetYaxis().SetTitle('Free Neutron Lifetime (s)')
    gr_free.GetYaxis().SetTitleOffset(1.65)
    gr_free.GetXaxis().SetTitle('Background Event Count')
    gr_free.Draw('AL')

    c1.Update()
    c1.SaveAs('{}/plot_lifetime_vs_background.free.pdf'.format(FIGURE_DIR))

    c2 = TCanvas('c2', 'c2', 600, 600)
    set_margin()
    gPad.SetLogx()
    gPad.SetGrid()
    set_graph_style(gr_bound)

    gr_bound.Draw('AL')
    gr_bound.GetYaxis().SetTitle('Bound Neutron Lifetime (year)')
    gr_bound.GetYaxis().SetTitleOffset(1.65)
    gr_bound.GetXaxis().SetTitle('Background Event Count')

    c2.Update()
    c2.SaveAs('{}/plot_lifetime_vs_background.bound.pdf'.format(FIGURE_DIR))

    input('Press any key to continue.')


def plot_ddt_exposure(filename, **kwargs):
    datetime_start = kwargs.get('datetime_start', datetime.datetime(2018, 7, 31))
    datetime_end = kwargs.get('datetime_end', datetime.datetime(2018, 9, 20))
    timestamp_start = datetime_start.timestamp()
    timestamp_end = datetime_end.timestamp()
    day_count = (datetime_end - datetime_start).days

    h_subrun_count = TH1D('h_subrun_count', 'h_subrun_count', day_count, timestamp_start, timestamp_end)
    h_exposure = TH1D('h_exposure', 'h_exposure', day_count, timestamp_start, timestamp_end)
    h_file_size = TH1D('h_file_size', 'h_file_size', day_count, timestamp_start, timestamp_end)
    h_event_count = TH1D('h_event_count', 'h_event_count', day_count, timestamp_start, timestamp_end)

    exposure = 0.
    subrun_start_times = []
    filename_samwebs = set()
    with open('{}/{}'.format(DATA_DIR, filename)) as f_json:
        for i, row in enumerate(f_json.readlines()):
            row = json.loads(row)
            try:
                event_count = row['Online.TotalEvents']
            except:
                print('event_count error')
                continue

            if event_count == 0:
                continue

            filename_samweb = row['file_name']
            if filename_samweb not in filename_samwebs:
                filename_samwebs.add(filename_samweb)
            else:
                continue

            subrun_start = row['Online.SubRunStartTime']
            subrun_end = row['Online.SubRunEndTime']
            file_size = row['file_size'] / 1.e9

            subrun_exposure = (subrun_end - subrun_start) / 60. / 60. / 24.
            exposure += subrun_exposure

            subrun_start_times.append(subrun_start)
            h_subrun_count.Fill(subrun_start)
            h_exposure.Fill(subrun_start, subrun_exposure)
            h_file_size.Fill(subrun_start, file_size)
            h_event_count.Fill(subrun_start, event_count)

    subrun_start_times = sorted(subrun_start_times)
    subrun_start_time_min = subrun_start_times[0]
    subrun_start_time_max = subrun_start_times[-1]
    print('first subrun: {}'.format(datetime.datetime.fromtimestamp(subrun_start_time_min).isoformat()))
    print('last subrun: {}'.format(datetime.datetime.fromtimestamp(subrun_start_time_max).isoformat()))
    print('exposure = {} days'.format(exposure))

    hists = [h_subrun_count, h_exposure, h_file_size, h_event_count]
    y_axis_title = ['File Count (/ Day)', 'Exposure Time (Day / Day)', 'File Size (GB / Day)', 'Event Count (/ Day)']

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()

    for i, hist in enumerate(hists):
        set_h1_style(hist)
        hist.GetXaxis().SetTimeDisplay(1)
        hist.GetXaxis().SetTimeFormat("%m/%d/%Y");
        hist.GetXaxis().SetTimeOffset(0, 'gmt')
        hist.GetYaxis().SetTitle(y_axis_title[i])
        hist.GetXaxis().SetTitle('Time (Month/Day/Year)')
        hist.Draw('hist')
        c1.Update()
        c1.SaveAs('{}/plot_ddt_exposure.{}.pdf'.format(FIGURE_DIR, hist.GetName()))
        input('Press any key to continue.')
        # break


def get_cosmic_ray_trigger_exposure():
    event_duration = 550.e-6    # s
    trigger_rate = 10.          # Hz

    exposure_per_second = event_duration * trigger_rate
    # exposure_per_year = (exposure_per_second * 60. * 60. * 24. * 365.) / (60. * 60. * 24.) # day
    exposure_per_year = exposure_per_second * 365. # day
    print('exposure_per_year = {} days'.format(exposure_per_year))


def plot_sensitivity_vs_efficiency_background():
    h1 = TH2D('h1', 'h1', 100, 0.5, 100.5, 100, 0, 1)

    for i in range(1, h1.GetNbinsX() + 1):
        for j in range(1, h1.GetNbinsY() + 1):
            background = h1.GetXaxis().GetBinCenter(i)
            efficiency = h1.GetYaxis().GetBinCenter(j)
            life_time = efficiency / background**0.5
            h1.SetBinContent(i, j, life_time)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogx()
    gPad.SetLogy()
    gPad.SetLogz()

    set_h2_style(h1)
    set_h2_color_style()

    h1.Draw('colz')
    h1.GetXaxis().SetTitle('Background Event Count')
    h1.GetYaxis().SetTitle('Efficiency')

    c1.Update()
    c1.SaveAs('{}/plot_sensitivity_vs_efficiency_background.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


# 20190604_nnbar_data_quality
gStyle.SetOptStat(0)
plot_ddt_exposure('get-metadata.all.json',
                  datetime_start=datetime.datetime(2018, 7, 31),
                  datetime_end=datetime.datetime(2019, 6, 4))

# gStyle.SetOptStat(0)
# plot_sensitivity_vs_efficiency_background()
# get_cosmic_ray_trigger_exposure()

# 20190220_nnbar_data_quality
# gStyle.SetOptStat(0)
# plot_ddt_exposure('get-metadata.json',
#                   datetime_start=datetime.datetime(2018, 7, 31),
#                   datetime_end=datetime.datetime(2019, 2, 20))

# 20181115_nnbar_data_quality
# gStyle.SetOptStat(0)
# plot_ddt_exposure('get-metadata.2018_11_15.json',
#                   datetime_start=datetime.datetime(2018, 7, 31),
#                   datetime_end=datetime.datetime(2018, 11, 15))

# 20180920_nnbar_exposure
# gStyle.SetOptStat(0)
# plot_ddt_exposure('get-metadata.json')

# 20180913_nnbar_prong
# gStyle.SetOptStat(0)
# plot_1d_cut('fProngCount3D', cosmic_filename='ddnnbar.hist.n_2000.root', signal_filename='genie_noscgen_10000_rndVertex.hist.n_2500.root', x_max=10, x_cut=4, tdirectory_signal='nnbarana', tdirectory_cosmic='neutronoscana', legend_text_cosmic='Cosmic Data', legend_text_signal='Signal MC', legend_fx1ndc=0.55)
# plot_1d_cut('fRecoHitGeV', cosmic_filename='ddnnbar.hist.n_2000.root', signal_filename='genie_noscgen_10000_rndVertex.hist.n_2500.root', x_max=3., x_cut=1.5, tdirectory_signal='nnbarana', tdirectory_cosmic='neutronoscana', legend_text_cosmic='Cosmic Data', legend_text_signal='Signal MC', legend_fx1ndc=0.55, rebin=5)
# plot_1d_cut('fRecoHitCount', cosmic_filename='ddnnbar.hist.n_2000.root', signal_filename='genie_noscgen_10000_rndVertex.hist.n_2500.root', x_cut=1.5, tdirectory_signal='nnbarana', tdirectory_cosmic='neutronoscana', legend_text_cosmic='Cosmic Data', legend_text_signal='Signal MC', legend_fx1ndc=0.55, rebin=10)
# plot_1d_cut('fExtentPlane', cosmic_filename='ddnnbar.hist.n_2000.root', signal_filename='genie_noscgen_10000_rndVertex.hist.n_2500.root', x_cut=1.5, tdirectory_signal='nnbarana', tdirectory_cosmic='neutronoscana', legend_text_cosmic='Cosmic Data', legend_text_signal='Signal MC', legend_fx1ndc=0.55, rebin=10, x_max=100)
# plot_1d_cut('fProngCount2D', cosmic_filename='ddnnbar.hist.n_2000.root', signal_filename='genie_noscgen_10000_rndVertex.hist.n_2500.root', x_max=20, x_cut=4, tdirectory_signal='nnbarana', tdirectory_cosmic='neutronoscana', legend_text_cosmic='Cosmic Data', legend_text_signal='Signal MC', legend_fx1ndc=0.55)

# 20180907_nnbar_background
# plot_lifetime_vs_background()

# 20180803_nnbar_first_data
# gStyle.SetOptStat(0)
# plot_daq_hit('raw.hist.n_20.root', draw_containment=True, draw_option='box')
# plot_daq_hit('fardet_r00030450_s00_DDNNBar.raw.hist.root', draw_containment=True, draw_option='colz')
# plot_daq_hit('raw.hist.n_1000.root', draw_containment=True, draw_option='box')
# plot_daq_hit('raw.hist.clean.root', draw_containment=True, draw_option='colz')
# plot_daq_hit_colz('raw.hist.root', draw_containment=True, draw_option='colz')
# plot_daq_hit('fardet_r00030450_s38_DDNNBar.raw.hist.root', draw_containment=True, draw_option='box')
# plot_daq_hit('fardet_r00030450_s38_DDNNBar.raw.hist.n_20.root', draw_containment=True, draw_option='box')
# plot_daq_hit('fardet_r00030450_s38_DDNNBar.raw.hist.n_30.root', draw_containment=True, draw_option='box')
# plot_daq_hit('fardet_r00030450_s38_DDNNBar.raw.hist.n_1000.root', draw_containment=True, draw_option='box')
# plot_daq_hit('fardet_r00030450_s04_DDNNBar.raw.hist.n_1000.root', draw_containment=True, draw_option='box')
# plot_daq_hit('fardet_r00030450_s05_DDNNBar.raw.hist.n_1000.root', draw_containment=True, draw_option='box')
# plot_1d_cut('fSliceHitCount',
#             cosmic_filename='raw.hist.root',
#             signal_filename='neutronosc_ddt_hist.maxCellCountFraction.clean.root',
#             x_max=400,
#             rebin=2,
#             legend_text_cosmic='Triggered Cosmic Data',
#             legend_text_signal='Signal MC',
#             legend_fx1ndc=0.40,
#             legend_fx2ndc=0.67,
#             x_cut=100)

# 20180731_doe
# get_nova_exposure()
# get_super_kamiokande_parameters()
# get_nova_parameters()
# get_life_time_bound(parameters=get_super_kamiokande_parameters(), make_plot=True, plot_suffix='.super_kamiokande')
# get_life_time_free(parameters=get_super_kamiokande_parameters(), make_plot=True, plot_suffix='.super_kamiokande')
# get_life_time_bound(parameters=get_nova_parameters(), make_plot=False)
# get_life_time_free(parameters=get_nova_parameters(), make_plot=False)
# plot_nova_sensitivity()
# parameters=get_super_kamiokande_parameters()
# parameters=get_nova_parameters()
# get_life_time_free(parameters=parameters, make_plot=True, plot_suffix='.nova', max_x=0.15)
# get_life_time_bound(parameters=parameters, make_plot=True, plot_suffix='.nova', max_x=100)

# 20180719_nnbar_globalconfig
# gStyle.SetOptStat(0)
# plot_one_planer('neutronosc_ddt_hist.no_hit_extent.cosmic.root')
# plot_feb_flasher('neutronosc_ddt_hist.flasher.root')
# calculate_efficiency()
# calculate_trigger_rate()
# print_trigger_evd_tex()
# plot_daq_hit('neutronosc_ddt_hist.maxCellCountFraction.cosmic.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.no_hit_extent.cosmic.large.root', draw_containment=True, draw_option='colz')
# plot_2d_cuts('fTrackWidthToLengthRatioXY',
#              cosmic_filename='neutronosc_ddt_hist.no_hit_extent.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.no_hit_extent.cosmic.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.01, y_max=1.01,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='-3.125 * pow(x - 0.4, 3)')
# plot_2d_cuts('fMultipleCellFractionXY',
#              cosmic_filename='neutronosc_ddt_hist.no_hit_extent.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.no_hit_extent.cosmic.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.01, y_max=1.01,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='0.1 * exp(-x / 0.05) + 0.02')
# plot_daq_hit('neutronosc_ddt_hist.no_hit_extent.cosmic.root', draw_containment=True, draw_option='colz')
# plot_2d_cuts('hPlaneCellExtentAsymmetry',
#              cosmic_filename='neutronosc_ddt_hist.no_hit_extent.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.no_hit_extent.cosmic.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.01, y_max=1.01,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              # equation='-0.6*x + 0.6'
#              equation='0.5')
# plot_daq_hit('neutronosc_ddt_hist.maxHitWidth.flasher.root', draw_containment=True, draw_option='colz')
# plot_daq_hit('neutronosc_ddt_hist.febEdgeCellCount_4.flasher.root', draw_containment=True, draw_option='colz')
# plot_daq_hit('neutronosc_ddt_hist.febEdgeCellCount_3.maxContinuousHitLength_15.root', draw_containment=True, draw_option='colz')
# plot_2d_cuts('hPlaneCellExtentAsymmetry',
#              cosmic_filename='neutronosc_ddt_hist.extent_asymmetry.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.extent_asymmetry.clean.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.01, y_max=1.01,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              # equation='-0.6*x + 0.6'
#              equation='0.5')
# plot_1d_cut('fFebEdgeCellCount',
#             # cosmic_filename='neutronosc_ddt_hist.febEdgeCellCount_4.cosmic.root',
#             cosmic_filename='neutronosc_ddt_hist.febEdgeCellCount_4.flasher.root',
#             signal_filename='neutronosc_ddt_hist.febEdgeCellCount_4.clean.root',
#             x_max=20,
#             y_max=0.5,
#             # log_y=True,
#             x_cut=5)

# 20180621_nnbar_topology
# gStyle.SetOptStat(0)
# plot_2d_cuts('hTrackLengthRatioXY',
#              cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.talk.clean.root',
#              log_x=False, log_y=False, log_z=False,
#              x_max=1.004, y_max=1.004,
#              x_cut=0.8, y_cut=0.8,
#              rebin=2,
#              cosmic_only=False,
#              grid=True)
# plot_2d_cuts('hHitDensityXY',
#              cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.talk.clean.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=1., y_max=1.,
#              x_cut=0.02, y_cut=0.02,
#              cosmic_only=False,
#              grid=True)
# plot_1d_cut('hTrackCount',
#             cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#             signal_filename='neutronosc_ddt_hist.talk.clean.root',
#             x_max=50,
#             log_y=True,
#             x_cut=0.05)
# plot_2d_cuts('hMomentOfInertiaXY',
#              cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.talk.clean.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=10000, y_max=10000,
#              x_cut=50, y_cut=50,
#              cosmic_only=False,
#              grid=True)
# plot_daq_hit('neutronosc_ddt_hist.talk.cosmic.2.root', draw_containment=True)
# calculate_efficiency()
# calculate_trigger_rate()
# plot_2d_cuts('fMultipleCellFractionXY',
#              cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.talk.clean.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=1.5, y_max=1.5,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='0.1 * exp(-x / 0.05) + 0.02')
# plot_2d_cuts('hPlaneCellExtentAsymmetry',
#              cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.talk.clean.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.01, y_max=1.01,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='-0.6*x + 0.6')
# plot_1d_cut('fXyAsymmetry', cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root', signal_filename='neutronosc_ddt_hist.talk.clean.root', y_max=0.25, x_max=1., rebin=5, x_cut=0.5)
# plot_2d_cuts('fTrackWidthToLengthRatioXY',
#              cosmic_filename='neutronosc_ddt_hist.talk.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.talk.clean.root',
#              log_x=False, log_y=False, log_z=False,
#              x_max=1., y_max=1.,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='-3.125 * pow(x - 0.4, 3)')
# plot_daq_hit_colz('neutronosc_ddt_hist.talk.clean.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.topology.root', draw_containment=False)
# plot_daq_hit('neutronosc_ddt_hist.width_length_ratio_xview.cosmic.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.asymmetry.cosmic.5.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.asymmetry.cosmic.4.root', draw_containment=True)
# slope = 0.6 / 1.
# intercept = 0.6
# equation = '-{}*x + {}'.format(slope, intercept)
# plot_2d_cuts('hPlaneCellExtentAsymmetry',
#              cosmic_filename='neutronosc_ddt_hist.asymmetry.cosmic.4.root',
#              signal_filename='neutronosc_ddt_hist.asymmetry.clean.2.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.5, y_max=1.5,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation=equation)
# plot_2d_cuts('fTrackWidthToLengthRatioXY',
#              cosmic_filename='neutronosc_ddt_hist.asymmetry.cosmic.4.root',
#              signal_filename='neutronosc_ddt_hist.asymmetry.clean.2.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.5, y_max=1.5,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='-3.125 * pow(x - 0.4, 3)')
# plot_2d_cuts('fMultipleCellFractionXY',
#              cosmic_filename='neutronosc_ddt_hist.asymmetry.cosmic.4.root',
#              signal_filename='neutronosc_ddt_hist.asymmetry.clean.2.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=1.5, y_max=1.5,
#              x_cut=0.5, y_cut=0.5,
#              rebin=None,
#              cosmic_only=False,
#              grid=True,
#              equation='0.1 * exp(-x / 0.05) + 0.02')
# plot_track('neutronosc_ddt_hist.high_stat.cosmic.root')
# plot_daq_hit('neutronosc_ddt_hist.high_stat.cosmic.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.no_asymmetry.cosmic.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.no_minHitExtentPlane.cosmic.root', draw_containment=True)
# plot_track('neutronosc_ddt_hist.width_to_length_ratio_0.2.cosmic.root')
# plot_daq_hit('neutronosc_ddt_hist.width_to_length_ratio_0.2.cosmic.root', draw_containment=True)
# plot_2d_cuts('fTrackWidthToLengthRatioXY',
#              cosmic_filename='neutronosc_ddt_hist.containment_y_min_4.cosmic.root',
#              signal_filename='neutronosc_ddt_hist.containment_y_min_4.cosmic.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=1.5, y_max=1.5,
#              x_cut=0.8, y_cut=0.8,
#              rebin=None,
#              cosmic_only=False,
#              grid=True)
# plot_track('neutronosc_ddt_hist.containment_y_min_4.cosmic.root')
# plot_daq_hit('neutronosc_ddt_hist.containment_y_min_4.cosmic.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.track_length_ratio_cosmic.4.root', draw_containment=True)
# plot_daq_hit('neutronosc_ddt_hist.fast.root', draw_containment=True)
# plot_track('neutronosc_ddt_hist.fast.root')
# plot_2d_cuts('hTrackLengthRatioXY',
#              cosmic_filename='neutronosc_ddt_hist.track_length_ratio_cosmic.3.root',
#              signal_filename='neutronosc_ddt_hist.track_length_ratio_clean.3.root',
#              log_x=False, log_y=False, log_z=True,
#              x_max=1.5, y_max=1.5,
#              x_cut=0.8, y_cut=0.8,
#              rebin=10,
#              cosmic_only=False,
#              grid=True)
# plot_track('neutronosc_ddt_hist.y_min_containment_cosmic.root')
# plot_track('neutronosc_ddt_hist.x_containment_cosmic.root')
# plot_track('neutronosc_ddt_hist.feb_flasher.root')
# plot_daq_hit('neutronosc_ddt_hist.feb_flasher.root')
# plot_2d_cuts('hMomentOfInertiaXY',
#              cosmic_filename='neutronosc_ddt_hist.moment_of_inertia_cosmic.root',
#              signal_filename='neutronosc_ddt_hist.moment_of_inertia_clean.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=10000, y_max=10000,
#              x_cut=50, y_cut=50,
#              cosmic_only=False)
# plot_2d_cuts('hHitDensityXY',
#              cosmic_filename='neutronosc_ddt_hist.hit_density_cosmic.root',
#              signal_filename='neutronosc_ddt_hist.hit_density_clean.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=1., y_max=1.,
#              x_cut=0.02, y_cut=0.02,
#              cosmic_only=False)
# plot_1d_cut('hTrackCount',
#             cosmic_filename='neutronosc_ddt_hist.hit_density_cosmic.root',
#             signal_filename='neutronosc_ddt_hist.hit_density_clean.root',
#             x_max=60,
#             log_y=True,
#             x_cut=0.05)

# 20180519_nnbar_blessing
# print_evd()

# 20180506_nnbar_containment
# gStyle.SetOptStat(0)
# plot_daq_hit('fill_tree_ymax_375_cosmic.root')
# plot_daq_hit('fill_tree_ymax_347_cosmic.root', draw_containment=True)
# calculate_efficiency()
# calculate_trigger_rate()
# plot_containment_effect()
# get_event_count('y_max_375_cosmic.log')
# check_dungs_topology_cut()
# plot_1d_cut('fMultipleCellFractionY',
#             cosmic_filename='MultipleCellFraction_occurrences_size_cosmic.root',
#             signal_filename='MultipleCellFraction_occurrences_size_clean.root',
#             x_cut=0.1)
# plot_1d_cut('fMultipleCellFractionAverage',
#             cosmic_filename='MultipleCellFraction_occurrences_size_cosmic.root',
#             signal_filename='MultipleCellFraction_occurrences_size_clean.root',
#             x_cut=0.05)
# plot_1d_cut('fMultipleCellFractionX',
#             cosmic_filename='MultipleCellFraction_occurrences_size_cosmic.root',
#             signal_filename='MultipleCellFraction_occurrences_size_clean.root',
#             x_cut=0.1)
# plot_2d_cuts('fMultipleCellFractionXY',
#              cosmic_filename='MultipleCellFraction_occurrences_size_cosmic.root',
#              signal_filename='MultipleCellFraction_occurrences_size_clean.root',
#              log_x=True, log_y=True, log_z=True,
#              x_max=1., y_max=1.,
#              x_cut=0.05, y_cut=0.05,
#              cosmic_only=False)

# 20180326_nnbar_width_length_ratio
# gStyle.SetOptStat(0)
# calculate_efficiency()
# plot_daq_hit_colz('random_vertex_clean.2.root')
# plot_daq_hit('trigger_cube_cut_cosmic.root')
# plot_1d_cut('fMaxHitExtentCell', cosmic_filename='cube_cut_cosmic.root', signal_filename='cube_cut_clean.root', y_max=0.12, rebin=5, x_cut=140)
# plot_1d_cut('fMinHitExtentPlane', cosmic_filename='cube_cut_cosmic.root', signal_filename='cube_cut_clean.root', y_max=0.15, x_max=80, rebin=1, x_cut=4)
# plot_1d_cut('fXyAsymmetry', cosmic_filename='cube_cut_cosmic.root', signal_filename='cube_cut_clean.root', y_max=0.14, x_max=2., rebin=4, x_cut=1.)
# plot_1d_cut('fSliceHitCount', cosmic_filename='cube_cut_cosmic.root', signal_filename='cube_cut_clean.root', y_max=0.14, x_max=400, rebin=5, x_cut=100)
# test_integral()
# plot_1d_cut('fMaxTrackLength', cosmic_filename='cube_cut_cosmic.root', signal_filename='cube_cut_clean.root', x_max=2000, y_max=0.13, rebin=5, x_cut=700)
# plot_2d_cuts('fTrackWidthToLengthRatioXY', cosmic_filename='cube_cut_cosmic.root', signal_filename='cube_cut_clean.root', log_x=False, log_y=False, log_z=False, x_max=1., y_max=1., cosmic_only=False)
# calculate_trigger_rate()
# calculate_efficiency()

# 20180301_nnbar_track
# gStyle.SetOptStat(0)
# plot_track('neutronosc_ddt_hist.track.root', dimension='2d', draw_track=False)
# plot_track_theta_variance()
# plot_slice_track_count()
# plot_1d_cut('fSliceHitCount', x_max=300, y_max=0.14, rebin=5, x_cut=100)
# plot_1d_cut('fXyAsymmetry', x_max=2., y_max=0.15, rebin=5, x_cut=1)
# plot_daq_hit_1d('neutronosc_ddt_hist.clean.root', 'fDaqHitYView')

# 20180128_nnbar_ddt_offline
# gStyle.SetOptStat(0)
# plot_daq_hit('neutronosc_ddt_hist.removeonedslices.root')
# plot_daq_hit('neutronosc_ddt_hist.containedslice.root')
# plot_daq_hit('neutronosc_ddt_hist.daqhit_count.root')
# plot_daq_hit('neutronosc_ddt_hist.xy_asymmetry.root')

# 20180103_nnbar_limit
# gStyle.SetOptStat('emr')
# plot(root_filename='nnbar_hist.root', hist_name='fSliceCount', x_min=-0.5, x_max=4.5, log_y=True)
# plot(root_filename='nnbar_hist.root', hist_name='fRecoHitGeV', x_min=-0.5, x_max=2.5, square_canvas=True, rebin=2)
# plot(root_filename='nnbar_hist.root', hist_name='fRecoHitCount', x_min=-0.5, x_max=140, square_canvas=True, rebin=2)
# plot(root_filename='nnbar_hist.root', hist_name='fExtentPlane', x_min=-0.5, x_max=100)
# plot(root_filename='nnbar_hist.root', hist_name='fExtentCellX', x_min=-0.5, x_max=200)
# plot(root_filename='nnbar_hist.root', hist_name='fExtentCellY', x_min=-0.5, x_max=200)
# plot(root_filename='nnbar_hist.root', hist_name='fFlsHitCount', square_canvas=True, rebin=10)
# plot(root_filename='nnbar_hist.root', hist_name='fFlsHitGeV', x_min=0.7, x_max=1.6, square_canvas=True)
# plot_life_time_bound()
# plot_life_time_free()
# print_evd()
