from rootalias import *
import scipy.constants
import csv
from datetime import datetime


FIGURE_DIR = '/Users/juntinghuang/beamer/20181011_light_yield_measurement/figures'
DATA_DIR = './data/scintillator'
# DATA_DIR = './data/calibration'


def plot_gain(filename):
    row_count = 0
    charges = []
    entries = []
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        reader = csv.reader(f_csv, delimiter='\t')
        next(reader)
        for row in reader:
            charge = -float(row[0]) / 50. # Coulomb
            entry = float(row[1])
            charges.append(charge)
            entries.append(entry)
            row_count += 1

    h1 = TH1D('h1', 'h1', row_count, charges[-1], charges[0])
    for i, charge in enumerate(charges):
        h1.Fill(charge, entries[i])

    # h1.Rebin(10)
    mean =h1.GetMean()
    sigma = h1.GetRMS()
    npe = (mean / sigma)**2.
    gain = mean / (npe * scipy.constants.elementary_charge)

    print('mean = {}'.format(mean))
    print('sigma = {}'.format(sigma))
    print('npe = {}'.format(npe))
    print('gain = {}'.format(gain))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)
    h1.Draw('hist')

    h1.GetXaxis().SetTitle('Charge (C)')
    h1.GetYaxis().SetTitle('Event Count')
    h1.GetYaxis().SetTitleOffset(1.5)
    # c1.Update()
    # draw_statbox(h1, x1= 0.65)

    # f1 = TF1('f1', 'gaus', 0., 20.e-12)
    # h1.Fit('f1')
    # f1.Draw('sames')

    t1 = TLatex()
    t1.SetNDC()
    t1.SetTextFont(43)
    t1.SetTextSize(28)
    t1.SetTextAlign(13)
    t1.DrawLatex(0.18, 0.86, 'mean = {:.1E} C'.format(mean))
    t1.DrawLatex(0.18, 0.8, '#sigma = {:.1E} C'.format(sigma))
    t1.DrawLatex(0.18, 0.74, 'NPE = {:.1f}'.format(npe))
    t1.DrawLatex(0.18, 0.68, 'gain = {:.1E}'.format(gain))

    c1.Update()
    c1.SaveAs('{}/plot_gain.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def get_npe_gain(filename):
    row_count = 0
    charges = []
    entries = []
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        reader = csv.reader(f_csv, delimiter='\t')
        next(reader)
        for row in reader:
            charge = -float(row[0]) / 50. # Coulomb
            entry = float(row[1])
            charges.append(charge)
            entries.append(entry)
            row_count += 1

    h1 = TH1D('h1', 'h1', row_count, charges[-1], charges[0])
    for i, charge in enumerate(charges):
        h1.Fill(charge, entries[i])

    mean =h1.GetMean()
    sigma = h1.GetRMS()
    npe = (mean / sigma)**2.
    gain = mean / (npe * scipy.constants.elementary_charge)

    return mean, sigma, npe, gain


def plot_gain_vs_hv():
    hvs = [500., 600., 700., 800., 900., 1000., 1100., 1150., 1200.]
    txts = ['F1ch200001.txt', 'F1ch200003.txt', 'F1ch200004.txt', 'F1ch200005.txt', 'F1ch200006.txt', 'F1ch200007.txt', 'F1ch200008.txt', 'F1ch200010.txt', 'F1ch200009.txt']

    gains = []
    for txt in txts:
        mean, sigma, npe, gain = get_npe_gain(txt)
        gains.append(gain)

    gr = TGraph(len(hvs), np.array(hvs), np.array(gains))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    gPad.SetLogy()
    set_graph_style(gr)
    gr.Draw('ALP')

    gr.GetXaxis().SetTitle('High Voltage (V)')
    gr.GetYaxis().SetTitle('Gain')

    c1.Update()
    c1.SaveAs('{}/plot_gain_hv.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_mean_per_pe_vs_hv():
    hvs = [500., 600., 700., 800., 900., 1000., 1100., 1150., 1200.]
    led_voltages = [1.550, 1.510, 1.500, 1.470, 1.460, 1.440, 1.430, 1.425, 1.420]
    txts = ['F1ch200001.txt', 'F1ch200003.txt', 'F1ch200004.txt', 'F1ch200005.txt', 'F1ch200006.txt', 'F1ch200007.txt', 'F1ch200008.txt', 'F1ch200010.txt', 'F1ch200009.txt']

    mean_per_pes = []
    for txt in txts:
        mean, sigma, npe, gain = get_npe_gain(txt)
        mean_per_pes.append(mean / npe)

    gr = TGraph(len(hvs), np.array(hvs), np.array(mean_per_pes))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    gPad.SetLogy()
    set_graph_style(gr)
    gr.Draw('AP')

    gr.GetXaxis().SetTitle('High Voltage (V)')
    gr.GetYaxis().SetTitle('Mean Charge per PE (C)')
    gr.GetYaxis().SetTitleOffset(1.5)

    c1.Update()
    c1.SaveAs('{}/plot_mean_per_pe_vs_hv.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_sigma2_vs_mu():
    # at 1100 V
    # led_voltages = [1.430, 1.435, 1.440, 1.445, 1.450, 1.460]
    # txts = ['F1ch200008.txt', 'F1ch200015.txt', 'F1ch200011.txt', 'F1ch200014.txt', 'F1ch200013.txt', 'F1ch200012.txt']

    # at 1000 V
    led_voltages = [1.480, 1.490, 1.500, 1.510, 1.520]
    txts = ['F1ch300007.txt', 'F1ch300008.txt', 'F1ch300009.txt', 'F1ch300010.txt', 'F1ch300011.txt']
    pedestal = get_npe_gain('F1ch300012.txt')[0]
    print('pedestal = {}'.format(pedestal))

    # means = []
    # sigma2s = []
    # for txt in txts:
    #     mean, sigma, npe, gain = get_npe_gain(txt)
    #     means.append(mean - pedestal)
    #     sigma2s.append(sigma**2)

    means = [9.587e-12, 1.678e-11, 2.8e-11, 4.6e-11, 7.335e-11]
    sigmas = [2.694e-12, 3.531e-12, 4.822e-12, 6.158e-12, 7.974e-12]
    sigma2s = [x**2 for x in sigmas]

    gr = TGraph(len(sigma2s), np.array(means), np.array(sigma2s))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_graph_style(gr)
    gr.Draw('AP')

    gr.GetXaxis().SetTitle('#mu (C)')
    gr.GetYaxis().SetTitle('#sigma^{2} (C^{2})')

    f1 = TF1('f1', 'pol1')
    gr.Fit('f1')
    intercept = f1.GetParameter(0)
    slope = f1.GetParameter(1)

    print('intercept = {}'.format(intercept))
    print('slope = {}'.format(slope))

    t1 = TLatex()
    t1.SetNDC()
    t1.SetTextFont(43)
    t1.SetTextSize(28)
    t1.SetTextAlign(13)
    t1.DrawLatex(0.25, 0.85, '#sigma^{{2}} = {:.1E} + {:.1E} #mu'.format(intercept, slope))

    c1.Update()
    c1.SaveAs('{}/plot_sigma2_vs_mu.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_npe_vs_led_voltage():
    hvs = [500., 600., 700., 800., 900., 1000., 1100., 1150., 1200.]
    led_voltages = [1.550, 1.510, 1.500, 1.470, 1.460, 1.440, 1.430, 1.425, 1.420]
    txts = ['F1ch200001.txt', 'F1ch200003.txt', 'F1ch200004.txt', 'F1ch200005.txt', 'F1ch200006.txt', 'F1ch200007.txt', 'F1ch200008.txt', 'F1ch200010.txt', 'F1ch200009.txt']

    npes = []
    for txt in txts:
        mean, sigma, npe, gain = get_npe_gain(txt)
        npes.append(npe)

    gr = TGraph(len(led_voltages), np.array(led_voltages), np.array(npes))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    gPad.SetLogy()
    set_graph_style(gr)
    gr.Draw('AP')

    gr.GetXaxis().SetTitle('LED Voltage (V)')
    gr.GetYaxis().SetTitle('NPE')
    gr.GetYaxis().SetRangeUser(1, 5000)

    c1.Update()
    c1.SaveAs('{}/plot_npe_vs_led_voltage.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_spectrum(filename, **kwargs):
    scale = kwargs.get('scale', None)

    row_count = 0
    charges = []
    entries = []
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        reader = csv.reader(f_csv, delimiter='\t')
        next(reader)
        for row in reader:
            charge = -float(row[0]) / 50. # Coulomb
            entry = float(row[1])

            if scale is not None:
                charge *= scale

            charges.append(charge)
            entries.append(entry)
            row_count += 1

    h1 = TH1D('h1', 'h1', row_count, charges[-1], charges[0])
    for i, charge in enumerate(charges):
        h1.Fill(charge, entries[i])

    return h1


def plot_spectrum(filename, **kwargs):
    rebin = kwargs.get('rebin', None)
    x_min = kwargs.get('x_min', None)
    x_max = kwargs.get('x_max', None)

    h1 = get_spectrum(filename)
    if rebin:
        h1.Rebin(rebin)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_h1_style(h1)

    spectrum = TSpectrum()
    spectrum.Search(h1, 3)
    spectrum.Print()
    poly_marker = h1.GetListOfFunctions().FindObject('TPolyMarker')

    h1.Draw('hist')
    if x_min is not None and x_max is not None:
        h1.GetXaxis().SetRangeUser(x_min, x_max)
    poly_marker.Draw()

    h1.GetXaxis().SetTitle('Charge (C)')
    h1.GetYaxis().SetTitle('Event Count')
    h1.GetYaxis().SetTitleOffset(1.5)
    # c1.Update()
    # draw_statbox(h1, x1= 0.65)

    # peak_xs = spectrum.GetPositionX()
    # peak_count = spectrum.GetNPeaks()
    # peaks = []
    # for i in range(peak_count):
    #     peaks.append(peak_xs[i])
    # peaks = sorted(peaks)
    # print('peaks = {}'.format(peaks))

    t1 = TLatex()
    t1.SetNDC()
    t1.SetTextFont(43)
    t1.SetTextSize(28)
    t1.SetTextAlign(13)

    # t1.DrawLatex(0.55, 0.85, 'Integral = {:.1E}'.format(h1.Integral()))
    # t1.DrawLatex(0.55, 0.78, 'Pedestal = {:.1E} C'.format(peaks[0]))
    # t1.DrawLatex(0.55, 0.71, 'Peak = {:.1E} C'.format(peaks[1]))

    c1.Update()
    c1.SaveAs('{}/plot_spectrum.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_spectra(**kwargs):
    calibration_constant = kwargs.get('calibration_constant', None)
    rebin = kwargs.get('rebin', None)
    # no_pedestal = kwargs.get('no_pedestal', False)

    filenames = ['F1ch300006.txt', 'F1ch300016.txt', 'F1ch300018.txt', 'F1ch300020.txt', 'F1ch300022.txt', 'F1ch300024.txt']
    filename_no_pedestals = ['F1ch300005.txt', 'F1ch300015.txt', 'F1ch300017.txt', 'F1ch300019.txt', 'F1ch300021.txt', 'F1ch300023.txt']
    legend_txts = ['NDOS', 'Production', 'Tanker', 'Tank 2', 'Tank 3', 'Tank 4']
    colors = [kBlack, kBlue, kRed + 1, kMagenta + 2, kGreen + 1, kOrange + 1]

    hists = []
    for i in range(len(filenames)):
        if calibration_constant is None:
            hist = get_spectrum(filenames[i])
        else:
            hist = get_spectrum(filenames[i], scale=1. / calibration_constant)
        if rebin:
            hist.Rebin(rebin)
        hist.Scale(1. / hist.GetMaximum())
        hists.append(hist)

    # peak_ys = []
    # for filename in filename_no_pedestals:
    #     peak_x, peak_y = get_spectrum_peak(filename, rebin=10)
    #     peak_ys.append(peak_y)
    # h1.Scale(1. / peak_ys[0])

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()

    lg1 = TLegend(0.58, 0.5, 0.92, 0.86)
    set_legend_style(lg1)

    for i, hist in enumerate(hists):
        set_h1_style(hist)
        hist.SetLineColor(colors[i])

        if i == 0:
            hist.Draw('hist')
            hist.GetYaxis().SetTitle('Event Count')
            hist.GetYaxis().SetTitleOffset(1.5)
            if calibration_constant is None:
                hist.GetXaxis().SetRangeUser(-1.e-11, 12.e-11)
                hist.GetXaxis().SetTitle('Charge (C)')
            else:
                hist.GetXaxis().SetRangeUser(-1.e-11 / calibration_constant, 12.e-11 / calibration_constant)
                hist.GetXaxis().SetTitle('NPE')
        else:
            hist.Draw('hist,sames')

        lg1.AddEntry(hist, legend_txts[i], 'l')

    lg1.Draw()

    c1.Update()
    if calibration_constant is None:
        c1.SaveAs('{}/plot_spectra.pdf'.format(FIGURE_DIR))
    else:
        c1.SaveAs('{}/plot_spectra.pe.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def print_event_rate():
    sample_names = [
        'NDOS',
        'Production',
        'Tanker',
        'Tank 2'
    ]
    filenames = [
        'F1ch300005.txt',
        'F1ch300015.txt',
        'F1ch300017.txt',
        'F1ch300019.txt'
    ]
    start_times = [
        datetime(2018, 10, 11, 18, 10),
        datetime(2018, 10, 12, 19, 58),
        datetime(2018, 10, 13, 12, 27),
        datetime(2018, 10, 14, 13, 16)
    ]
    end_times = [
        datetime(2018, 10, 12, 10, 10),
        datetime(2018, 10, 13, 11, 57),
        datetime(2018, 10, 14, 12, 51),
        datetime(2018, 10, 15, 10, 9)
    ]

    durations = []              # minutes
    for i in range(len(start_times)):
        durations.append((end_times[i] - start_times[i]).total_seconds() / 60.)

    event_counts = []
    for filename in filenames:
        h1 = get_spectrum(filename)
        event_count = h1.Integral()
        event_counts.append(event_count)

    event_rates = []
    for i, duration in enumerate(durations):
        event_rates.append(event_counts[i] / durations[i])

    for i, event_rate in enumerate(event_rates):
        print('{} & {:.1f} & {:.0f} & {:.0f} \\\\'.format(sample_names[i], durations[i] / 60., event_counts[i], event_rates[i]))


def print_peaks():
    filenames = [
        'F1ch300005.txt',
        'F1ch300015.txt',
        'F1ch300017.txt',
        'F1ch300019.txt'
    ]

    peak_xs = []
    peak_ys = []
    for filename in filenames:
        peak_x, peak_y = get_spectrum_peak(filename, rebin=10)
        peak_xs.append(peak_x)
        peak_ys.append(peak_y)

    for peak_x in peak_xs:
        print('peak_x = {:.2E}'.format(peak_x))
        print('peak_x = {:.1F}'.format(peak_x / 8.854658242290205e-13))

def get_spectrum_peak(filename, **kwargs):
    rebin = kwargs.get('rebin', None)
    h1 = get_spectrum(filename)
    if rebin:
        h1.Rebin(rebin)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_h1_style(h1)

    spectrum = TSpectrum()
    spectrum.Search(h1, 3)
    # spectrum.Print()
    # poly_marker = h1.GetListOfFunctions().FindObject('TPolyMarker')
    peak_xs = spectrum.GetPositionX()
    peak_ys = spectrum.GetPositionY()
    peak_count = spectrum.GetNPeaks()
    # peaks = []
    # for i in range(peak_count):
    #     peaks.append(peak_xs[i])
    # peaks = sorted(peaks)
    # print('peaks = {}'.format(peaks))
    return peak_xs[0], peak_ys[0]


# 20181005_testbeam_light_yield_setup
gStyle.SetOptStat(0)
# gStyle.SetOptStat('mri')
# gStyle.SetOptFit(0)
# plot_gain('F1ch200008.txt')
# plot_gain('F1ch200015.txt')
# plot_gain('F1ch200011.txt')
# plot_gain_vs_hv()
# plot_mean_per_pe_vs_hv()
# plot_sigma2_vs_mu()
# plot_npe_vs_led_voltage()
# plot_spectrum('F1ch300000.txt')
# plot_spectrum('F1ch300001.txt', rebin=5, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300002.txt')
# plot_spectra('F1ch300001.txt', 'F1ch300002.txt')
# plot_spectrum('F1ch300005.txt', rebin=5, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300006.txt', rebin=5, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300016.txt', rebin=5, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300018.txt', rebin=5, x_min=-0.02e-9, x_max=0.15e-9)
# plot_sigma2_vs_mu()
# plot_gain('F1ch300007.txt')
# plot_gain('F1ch300008.txt')
# plot_gain('F1ch300009.txt')
# plot_gain('F1ch300010.txt')
# plot_gain('F1ch300011.txt')
# plot_gain('F1ch300012.txt')
calibration_constant = 8.854658242290205e-13 # C / PE
plot_spectra(rebin=10, calibration_constant=calibration_constant)
# plot_spectra(rebin=10)
# no pedestal
# plot_spectrum('F1ch300005.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300015.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300017.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300019.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# added pedestal
# plot_spectrum('F1ch300006.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300016.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300018.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# plot_spectrum('F1ch300020.txt', rebin=10, x_min=-0.02e-9, x_max=0.15e-9)
# print_event_rate()
# get_spectrum_peak('F1ch300005.txt', rebin=10)
# get_spectrum_peak('F1ch300015.txt', rebin=10)
# get_spectrum_peak('F1ch300017.txt', rebin=10)
# get_spectrum_peak('F1ch300019.txt', rebin=10)
# print_peaks()
