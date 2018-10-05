from rootalias import *
import scipy.constants
import csv

FIGURE_DIR = './figures'

def plot_gain():
    row_count = 0
    charges = []
    entries = []
    with open('data/F1ch200000.txt') as f_csv:
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
    h1.GetXaxis().SetRangeUser(11.5e-12, 14.5e-12)
    h1.GetYaxis().SetTitleOffset(1.5)
    c1.Update()
    draw_statbox(h1, x1= 0.65)

    t1 = TLatex()
    t1.SetTextFont(43)
    t1.SetTextSize(28)
    # t1.SetTextAlign(13)
    t1.DrawLatex(1.165e-11, 4800, 'NPE = {:.1f}'.format(npe))
    t1.DrawLatex(1.165e-11, 4300, 'gain = {:.1E}'.format(gain))

    c1.Update()
    c1.SaveAs('{}/light_yield.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')

gStyle.SetOptStat('mr')
plot_gain()
