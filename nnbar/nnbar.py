from rootalias import *
import math
import numpy as np

figure_dir = 'figures'

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


# plot_life_time_bound()
plot_life_time_free()
