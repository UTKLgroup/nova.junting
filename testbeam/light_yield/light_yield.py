from rootalias import *
import scipy.constants
import csv


FIGURE_DIR = '/Users/juntinghuang/beamer/20181005_testbeam_light_yield_setup/figures'
DATA_DIR = './data'


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

    h1.Rebin(10)
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

    t1 = TLatex()
    t1.SetTextFont(63)
    t1.SetTextSize(28)
    t1.SetTextAlign(13)
    t1.DrawLatex(0.5, 0.5, 'NPE = {:.1f}'.format(npe))

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
    led_voltages = [1.430, 1.435, 1.440, 1.445, 1.450, 1.460]
    txts = ['F1ch200008.txt', 'F1ch200015.txt', 'F1ch200011.txt', 'F1ch200014.txt', 'F1ch200013.txt', 'F1ch200012.txt']

    means = []
    sigma2s = []
    for txt in txts:
        mean, sigma, npe, gain = get_npe_gain(txt)
        means.append(mean)
        sigma2s.append(sigma**2)

    gr = TGraph(len(sigma2s), np.array(means), np.array(sigma2s))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_graph_style(gr)
    gr.Draw('AP')

    gr.GetXaxis().SetTitle('#mu (C)')
    gr.GetYaxis().SetTitle('#sigma^{2} (C^{2})')
    gr.Fit('pol1')

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


# 20181005_testbeam_light_yield_setup
gStyle.SetOptStat(0)
# plot_gain('F1ch200008.txt')
# plot_gain('F1ch200015.txt')
# plot_gain('F1ch200011.txt')
# plot_gain_vs_hv()
# plot_mean_per_pe_vs_hv()
# plot_sigma2_vs_mu()
plot_npe_vs_led_voltage()
