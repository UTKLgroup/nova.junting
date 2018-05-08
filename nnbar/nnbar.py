from rootalias import *
import math
import numpy as np
import random

FIGURE_DIR = '/Users/juntinghuang/beamer/20180506_nnbar_containment/figures'
DATA_DIR = './data'

exposure_0 = 2.45               # 1.e34 * neutron * year
efficiency_0 = 12.1e-2
background_0 = 24.1
suppression_factor_0 = 0.517    # 1.e23 s-1
exposure_sigma = 3.e-2 * exposure_0
efficiency_sigma = 22.9e-2 * efficiency_0
background_sigma = 23.7e-2 * background_0
suppression_factor_sigma = 0.3 * suppression_factor_0
event_count_observe = 24
second_in_year = 3.16           # 1.e7 s / year

colors = [
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


def plot_life_time_bound():
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

    life_time_cl = 1. / event_rate_true_cl
    print('event_rate_true_cl = ', event_rate_true_cl * 1.e-34, ' per year at 90% C.L.')
    print('life_time_cl = ', life_time_cl * 1.e34, ' years at 90% C.L.')

    probabilities = list(map(lambda x: x / total_area, probabilities))
    gr = TGraph(len(event_rate_trues), event_rate_trues, np.array(probabilities))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_graph_style(gr)
    set_margin()

    gr.Draw('AL')
    gr.GetYaxis().SetDecimals()
    gr.GetXaxis().SetTitle('True Event Count per 10^{34} neutron year')
    gr.GetYaxis().SetTitle('Probability Density')
    gr.GetXaxis().SetRangeUser(0, max(event_rate_trues))
    gr.GetYaxis().SetTitleOffset(1.5)
    max_y = 0.03
    gr.GetYaxis().SetRangeUser(0, max_y)
    tl = TLine(event_rate_true_cl, 0, event_rate_true_cl, max_y)
    tl.SetLineWidth(3)
    tl.SetLineColor(kRed)
    tl.SetLineStyle(7)
    tl.Draw()

    lg1 = TLegend(0.6, 0.8, 0.75, 0.88)
    set_legend_style(lg1)
    lg1.AddEntry(tl, '90% C.L. Limit', 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_life_time_bound.pdf'.format(figure_dir))
    input('Press any key to continue.')


def plot_life_time_free():
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
                        event_rate_true = inverse_square_free_life_time_true / suppression_factor * second_in_year * 1.e2
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

    print('inverse_square_free_life_time_true_cl = ', inverse_square_free_life_time_true_cl, ' at 90% C.L.')
    print('free_life_time_cl = ', (1. / inverse_square_free_life_time_true_cl)**0.5, ' * 1e8 secondat 90% C.L.')

    probabilities = list(map(lambda x: x / total_area, probabilities))
    gr = TGraph(len(inverse_square_free_life_time_trues), inverse_square_free_life_time_trues, np.array(probabilities))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_graph_style(gr)
    set_margin()

    gr.Draw('AL')
    gr.GetYaxis().SetDecimals()
    gr.GetXaxis().SetTitle('1 / #tau^{2} (10^{-16} s^{-2})')
    gr.GetYaxis().SetTitle('Probability Density')
    gr.GetXaxis().SetRangeUser(0, max(inverse_square_free_life_time_trues))
    gr.GetYaxis().SetTitleOffset(1.5)
    gr.GetYaxis().SetNdivisions(505, 1)
    gr.GetXaxis().SetNdivisions(505, 1)
    max_y = 19
    gr.GetYaxis().SetRangeUser(0, max_y)
    gr.GetYaxis().SetRangeUser(0, max_y)
    tl = TLine(inverse_square_free_life_time_true_cl, 0, inverse_square_free_life_time_true_cl, max_y)
    tl.SetLineWidth(3)
    tl.SetLineColor(kRed)
    tl.SetLineStyle(7)
    tl.Draw()

    lg1 = TLegend(0.6, 0.8, 0.75, 0.88)
    set_legend_style(lg1)
    lg1.AddEntry(tl, '90% C.L. Limit', 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_life_time_free.pdf'.format(figure_dir))
    input('Press any key to continue.')


def print_evd():
    events = {
        1: 'p + \\bar{n} \\rightarrow \\pi^{+} + \\pi^{0}',
        2: 'p + \\bar{n} \\rightarrow \\pi^{+} + 2\\pi^{0}',
        3: 'p + \\bar{n} \\rightarrow \\pi^{+} + 3\\pi^{0}',
        4: 'p + \\bar{n} \\rightarrow 2\\pi^{+} + \\pi^{-} + \\pi^{0}',
        5: 'p + \\bar{n} \\rightarrow 2\\pi^{+} + \\pi^{-} + 2\\pi^{0}',
        6: 'p + \\bar{n} \\rightarrow 2\\pi^{+} + \\pi^{-} + 2\\omega^{0}',
        7: 'p + \\bar{n} \\rightarrow 3\\pi^{+} + 2\\pi^{-} + \\pi^{0}',
        8: 'n + \\bar{n} \\rightarrow \\pi^{+} + \\pi^{-}',
        9: 'n + \\bar{n} \\rightarrow 2\\pi^{0}',
        10: 'n + \\bar{n} \\rightarrow \\pi^{+} + \\pi^{-} + \\pi^{0}',
        11: 'n + \\bar{n} \\rightarrow \\pi^{+} + \\pi^{-} + 2\\pi^{0}',
        12: 'n + \\bar{n} \\rightarrow \\pi^{+} + \\pi^{-} + 3\\pi^{0}',
        13: 'n + \\bar{n} \\rightarrow 2\\pi^{+} + 2\\pi^{-}',
        14: 'n + \\bar{n} \\rightarrow 2\\pi^{+} + 2\\pi^{-} + \\pi^{0}',
        15: 'n + \\bar{n} \\rightarrow \\pi^{+} + \\pi^{-} + \\omega^{0}',
        16: 'n + \\bar{n} \\rightarrow 2\\pi^{+} + 2\\pi^{-} + 2\\pi^{0}'
    }

    with open('print_evd.tex', 'w') as f_evd:
        for i in range(1, 17):
            reaction = events[i]
            f_evd.write('\\begin{frame}\n')
            f_evd.write('  \\frametitle{{Example of ${}$}}\n'.format(reaction))
            f_evd.write('  \\begin{figure}\n')
            f_evd.write('    \\includegraphics[scale = 0.25]{{figures/{{nnbar.{}}}.png}}\n'.format(i))
            f_evd.write('    \\caption{{An example event of ${}$.}}\n'.format(reaction))
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


def plot_daq_hit(filename):
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
    h_x.Draw('box')

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

    c1.Update()
    c1.SaveAs('{}/plot_daq_hit.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_daq_hit_colz(filename):
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

    lx_b = TLine(3, 3, 891, 3)
    lx_t = TLine(3, 380, 891, 380)
    lx_l = TLine(3, 3, 3, 380)
    lx_r = TLine(891, 3, 891, 380)
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

    ly_b = TLine(3, 3, 891, 3)
    ly_t = TLine(3, 375, 891, 375)
    ly_l = TLine(3, 3, 3, 375)
    ly_r = TLine(891, 3, 891, 375)
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

        color = random.choice(colors)
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

        color = random.choice(colors)
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
    legend_left = kwargs.get('legend_left', False)

    tf_cosmic = TFile('{}/{}'.format(DATA_DIR, cosmic_filename))
    tf_clean = TFile('{}/{}'.format(DATA_DIR, signal_filename))

    h_cosmic = tf_cosmic.Get('neutronoscana/{}'.format(hist_name))
    h_clean = tf_clean.Get('neutronoscana/{}'.format(hist_name))

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
    if y_max:
        h_cosmic.GetYaxis().SetRangeUser(0, y_max)

    set_h1_style(h_clean)
    h_clean.SetLineColor(kRed + 2)
    h_clean.Draw('hist,sames')

    lg1 = None
    if legend_left:
        lg1 = TLegend(0.21, 0.7, 0.48, 0.85)
    else:
        lg1 = TLegend(0.57, 0.7, 0.84, 0.85)
    set_legend_style(lg1)
    lg1.AddEntry(h_cosmic, 'Cosmic ray', 'l')
    lg1.AddEntry(h_clean, 'Signal', 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_1d_cut.{}.{}.pdf'.format(FIGURE_DIR, cosmic_filename, hist_name))
    input('Press any key to continue.')


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

    h_clean.Scale(1. / h_clean.Integral())
    h_cosmic.Scale(1. / h_cosmic.Integral())

    # for width-length ratio
    # x0 = 0.4
    # y0 = 0.12
    # equation = '{} * pow(x - {}, 3)'.format(-y0 / x0**3, x0)
    # # -1.875 * pow(x - 0.4, 3)
    # print('equation = {}'.format(equation))
    # tf1 = TF1('tf1', equation, 0, x0)

    # for multiple cell fraction
    # equation = '0.05 * exp(-x / 0.05) + 0.05 * exp(-x / 0.2)'
    # equation = '0.08 * exp(-x / 0.03) + 0.01'
    # equation = '0.1 * exp(-x / 0.05) + 0.01'
    equation = '0.1 * exp(-x / 0.05) + 0.02'
    # equation = '0.11 * exp(-x / 0.05) + 0.02'
    print('equation = {}'.format(equation))
    tf1 = TF1('tf1', equation, 0, 1)

    tf1.SetLineWidth(3)
    tf1.SetLineColor(kRed)

    if not signal_only:
        c1 = TCanvas('c1', 'c1', 600, 600)
        set_margin()
        gPad.SetRightMargin(0.2)
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
        tf1.Draw('sames')
        c1.Update()
        c1.SaveAs('{}/plot_2d_cuts.{}.{}.cosmic.pdf'.format(FIGURE_DIR, cosmic_filename, histname))

    if not cosmic_only:
        c2 = TCanvas('c2', 'c2', 600, 600)
        set_margin()
        gPad.SetRightMargin(0.2)
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
    event_count = 1636.
    event_duration = 0.55e-3    # s
    exposure = event_count * event_duration

    cut_names = ['pre-containment', 'containment', 'width-length ratio', 'max track length', 'cell hit count', 'XY view asymmetry', 'hit extent']
    slice_counts = [106904, 12212., 185., 101., 79., 65., 61.]

    fractions = [slice_count / slice_counts[0] for slice_count in slice_counts]
    rates = [slice_count / exposure for slice_count in slice_counts]
    for i, fraction in enumerate(fractions):
        rate = '{:.0f}'.format(rates[i])
        print('{} & {:.0f} & \SI{{{}}}{{}} & \SI{{{}}}{{}} \\\\'.format(cut_names[i], slice_counts[i], '{:.1E}'.format(fractions[i]), rate))


def calculate_efficiency():
    event_count = 1636.
    event_duration = 0.55e-3    # s
    exposure = event_count * event_duration

    cut_names = ['pre-containment', 'containment', 'width-length ratio', 'max track length', 'cell hit count', 'XY view asymmetry', 'hit extent']
    # slice_counts = [5157., 5157., 4589., 4588., 4573., 4562., 4539.]
    slice_counts = [2673., 2108., 1872., 1870., 1855., 1849., 1844.]

    fractions = [slice_count / slice_counts[0] for slice_count in slice_counts]
    rates = [slice_count / exposure for slice_count in slice_counts]
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

# 20180506_nnbar_containment
gStyle.SetOptStat(0)
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
plot_2d_cuts('fMultipleCellFractionXY',
             cosmic_filename='MultipleCellFraction_occurrences_size_cosmic.root',
             signal_filename='MultipleCellFraction_occurrences_size_clean.root',
             log_x=True, log_y=True, log_z=True,
             x_max=1., y_max=1.,
             x_cut=0.05, y_cut=0.05,
             cosmic_only=False)

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
