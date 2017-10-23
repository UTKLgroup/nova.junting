from subprocess import call
from rootalias import *

figure_dir = '/Users/juntinghuang/beamer/20171022_tdslicer_nd_genie/figures'
data_dir = 'data/nd_genie'

def plot(**kwargs):
    hist_name = kwargs.get('hist_name', 'NumSlices')
    x_min = kwargs.get('x_min')
    x_max = kwargs.get('x_max')
    log_y = kwargs.get('log_y', False)
    log_x = kwargs.get('log_x', False)
    x_title = kwargs.get('x_title')
    y_title = kwargs.get('y_title')
    rebin = kwargs.get('rebin')
    statbox_position = kwargs.get('statbox_position', 'right')
    root_filename = kwargs.get('root_filename')

    f_slicer = TFile('{}/{}'.format(data_dir, root_filename))
    h_4d = f_slicer.Get('slicerana/{}'.format(hist_name))
    h_td = f_slicer.Get('tdslicerana/{}'.format(hist_name))

    if rebin:
        h_4d.Rebin(rebin)
        h_td.Rebin(rebin)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy(log_y)
    gPad.SetLogx(log_x)

    set_h1_style(h_4d)
    h_4d.SetName('slicer4d')
    if x_min and x_max:
        h_4d.GetXaxis().SetRangeUser(x_min, x_max)
        h_td.GetXaxis().SetRangeUser(x_min, x_max)
    if not log_y:
        h_4d.GetYaxis().SetRangeUser(0, get_max_y([h_4d, h_td]) * 1.1)
    if x_title:
        h_4d.GetXaxis().SetTitle(x_title)
    if y_title:
        h_4d.GetYaxis().SetTitle(y_title)
    h_4d.Draw()

    set_h1_style(h_td)
    h_td.SetName('tdslicer')
    h_td.SetLineColor(kRed)
    h_td.Draw('sames')

    c1.Update()
    draw_statboxes(h_td, h_4d, position=statbox_position)

    c1.Update()
    c1.SaveAs('{}/plot.{}.{}.pdf'.format(figure_dir, root_filename, hist_name))
    input('Press any key to continue.')


def plot_slice_matching(tree_name, root_filename):
    f_slicer = TFile('{}/{}'.format(data_dir, root_filename))
    tolerance = 6

    z_scale = 27.0
    t_scale = 57.0
    # z_scale = 100.0
    # t_scale = 10.0
    h_distance = TH1D('h_distance', 'h_distance', 100, 0, 50)
    h_delta_min_z = TH1D('h_delta_min_z', 'h_delta_min_z', 100, 0, 50)
    h_delta_max_z = TH1D('h_delta_max_z', 'h_delta_max_z', 100, 0, 50)
    h_delta_mean_tns = TH1D('h_delta_mean_tns', 'h_delta_mean_tns', 100, 0, 50)

    # z_scale = 1.0
    # t_scale = 1.0
    # h_distance = TH1D('h_distance', 'h_distance', 100, 0, 1000)
    # h_delta_min_z = TH1D('h_delta_min_z', 'h_delta_min_z', 100, 0, 1000)
    # h_delta_max_z = TH1D('h_delta_max_z', 'h_delta_max_z', 100, 0, 1000)
    # h_delta_mean_tns = TH1D('h_delta_mean_tns', 'h_delta_mean_tns', 100, 0, 1000)

    for cluster in f_slicer.Get('slicerana/{}'.format(tree_name)):
        delta_min_z= abs(cluster.xViewMinZ - cluster.yViewMinZ) / z_scale;
        delta_max_z= abs(cluster.xViewMaxZ - cluster.yViewMaxZ) / z_scale;
        delta_mean_tns = abs(cluster.xViewMeanTNS - cluster.yViewMeanTNS) / t_scale;
        distance = (delta_min_z**2 + delta_max_z**2 + delta_mean_tns**2)**0.5
        h_delta_min_z.Fill(delta_min_z)
        h_delta_max_z.Fill(delta_max_z)
        h_delta_mean_tns.Fill(delta_mean_tns)
        h_distance.Fill(distance)

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetLogy()

    set_h1_style(h_delta_min_z)
    set_h1_style(h_delta_max_z)
    set_h1_style(h_delta_mean_tns)
    h_delta_max_z.SetLineColor(kGreen + 2)
    h_delta_mean_tns.SetLineColor(kRed)
    h_delta_min_z.Draw()
    h_delta_max_z.Draw('sames')
    h_delta_mean_tns.Draw('sames')
    h_delta_min_z.GetXaxis().SetTitle('Difference between X and Y Views')
    h_delta_min_z.GetYaxis().SetTitle('Slice Count')
    h_delta_min_z.GetYaxis().SetTitleOffset(1.5)
    h_delta_min_z.SetName('MinZ')
    h_delta_max_z.SetName('MaxZ')
    h_delta_mean_tns.SetName('MeanTNS')
    c1.Update()
    draw_statboxess(h_delta_min_z, h_delta_max_z, h_delta_mean_tns)
    c1.Update()
    c1.SaveAs('{}/plot_slice_matching.{}.{}.h_delta.{}.{}.pdf'.format(figure_dir, root_filename, tree_name, z_scale, t_scale))

    c2 = TCanvas('c2', 'c2', 600, 600)
    set_margin()
    set_margin()
    gPad.SetLogy()
    set_h1_style(h_distance)
    y_max = h_distance.GetMaximum() * 1.1
    h_distance.GetXaxis().SetTitle('Difference between X and Y Views')
    h_distance.GetYaxis().SetTitle('Slice Count')
    h_distance.GetYaxis().SetTitleOffset(1.5)
    h_distance.GetYaxis().SetRangeUser(1, y_max)
    h_distance.SetName('D_{XY}')
    h_distance.Draw()

    l1 = TLine(tolerance, 0, tolerance, y_max)
    l1.SetLineColor(kRed)
    l1.SetLineStyle(2)
    l1.SetLineWidth(3)
    l1.Draw()
    c2.Update()
    draw_statbox(h_distance)

    text = TLatex(7, 10, 'D_{XY} = 6')
    text.SetTextFont(43);
    text.SetTextSize(35);
    text.SetTextColor(kRed);
    text.Draw()

    c2.Update()
    c2.SaveAs('{}/plot_slice_matching.{}.{}.h_distance.{}.{}.pdf'.format(figure_dir, root_filename, tree_name, z_scale, t_scale))
    raw_input('Press any key to continue.')


def plot_tuned(**kwargs):
    hist_name = kwargs.get('hist_name', 'NumSlices')
    x_min = kwargs.get('x_min')
    x_max = kwargs.get('x_max')
    log_y = kwargs.get('log_y', False)
    log_x = kwargs.get('log_x', False)
    statbox_position = kwargs.get('statbox_position', 'right')
    root_filename =  kwargs.get('root_filename', 'fd_genie.nonswap.root')
    root_filename_new =  kwargs.get('root_filename_new', 'fd_genie.nonswap.new.root')

    f_slicer = TFile('{}/{}'.format(data_dir, root_filename))
    f_slicer_new = TFile('{}/{}'.format(data_dir, root_filename_new))

    h_4d = f_slicer.Get('slicerana/{}'.format(hist_name))
    h_td = f_slicer.Get('tdslicerana/{}'.format(hist_name))
    h_td_new = f_slicer_new.Get('tdslicerana/{}'.format(hist_name))
    h_true = f_slicer_new.Get('trueslicerana/{}'.format(hist_name))

    gStyle.SetOptStat('nemr')
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy(log_y)
    gPad.SetLogx(log_x)

    h_axis = TH1D('h_axis', 'h_axis', 121,-0.5,120.5)
    if x_min and x_max:
        h_axis.GetXaxis().SetRangeUser(x_min, x_max)
    if not log_y:
        h_axis.GetYaxis().SetRangeUser(0, get_max_y([h_4d, h_td, h_td_new]) * 1.1)
    set_h1_style(h_axis)
    h_axis.SetStats(0)
    h_axis.GetXaxis().SetTitle(h_4d.GetXaxis().GetTitle())
    h_axis.GetYaxis().SetTitle(h_4d.GetYaxis().GetTitle())
    h_axis.Draw()

    set_h1_style(h_4d)
    h_4d.SetLineColor(kBlack)
    h_4d.SetName('slicer4d')
    h_4d.Draw('sames')

    set_h1_style(h_true)
    h_true.SetLineColor(kMagenta + 1)
    h_true.SetName('trueslicer')
    h_true.Draw('sames')

    set_h1_style(h_td)
    h_td.SetName('tdslicer')
    h_td.Draw('sames')

    set_h1_style(h_td_new)
    h_td_new.SetName('tuned tdslicer')
    h_td_new.SetLineColor(kRed)
    h_td_new.Draw('sames')

    c1.Update()
    draw_statboxesss(h_td_new, h_td, h_4d, h_true)

    c1.Update()
    c1.SaveAs('{}/plot_tuned.{}.{}.pdf'.format(figure_dir, root_filename_new, hist_name))
    raw_input('Press any key to continue.')


def plot_ratio(**kwargs):
    hist_name = kwargs.get('hist_name', 'NumSlices')
    x_min = kwargs.get('x_min')
    x_max = kwargs.get('x_max')
    log_y = kwargs.get('log_y', False)
    log_x = kwargs.get('log_x', False)
    root_filename =  kwargs.get('root_filename', 'fd_genie.nonswap.root')
    root_filename_new =  kwargs.get('root_filename_new', 'fd_genie.nonswap.new.root')

    f_slicer = TFile('{}/{}'.format(data_dir, root_filename))
    f_slicer_new = TFile('{}/{}'.format(data_dir, root_filename_new))

    h_td = f_slicer.Get('tdslicerana/{}'.format(hist_name))
    h_td_new = f_slicer_new.Get('tdslicerana/{}'.format(hist_name))

    gStyle.SetOptStat(0)
    c1 = TCanvas('c1', 'c1', 800, 600)
    gPad.SetBottomMargin(0.15)
    gPad.SetLeftMargin(0.15)

    pad1 = TPad("pad1", "pad1", 0, 0.4, 1, 1)
    pad1.SetTopMargin(0.15)
    pad1.SetBottomMargin(0)
    pad1.SetLeftMargin(0.15)
    pad1.Draw()
    pad1.cd()
    if log_y:
        gPad.SetLogy()
    if log_x:
        gPad.SetLogx()
    set_h1_style(h_td_new)
    h_td_new.SetLineColor(kRed)
    h_td_new.SetMarkerColor(kRed)
    h_td_new.GetYaxis().SetTitle('Y Title')
    if not log_y:
        h_td_new.GetYaxis().SetRangeUser(0, max(h_td_new.GetMaximum(), h_td.GetMaximum()) * 1.2)
    h_td_new.GetYaxis().SetTitleOffset(1.5)
    h_td_new.Draw("hist")
    set_h1_style(h_td)
    h_td.Draw("sames, hist")
    # lg1 = TLegend(0.7, 0.5, 0.9, 0.8)
    lg1 = TLegend(0.6, 0.5, 0.9, 0.8)
    set_legend_style(lg1)
    lg1.AddEntry(h_td_new, 'h_td_new', 'l')
    lg1.AddEntry(h_td, 'h_td', 'l')
    lg1.Draw()

    c1.cd()
    pad2 = TPad('pad2', 'pad2', 0, 0, 1, 0.4)
    pad2.SetTopMargin(0)
    pad2.SetLeftMargin(0.15)
    pad2.SetBottomMargin(0.4)
    pad2.Draw()
    pad2.cd()
    gPad.SetGrid()
    h_ratio = h_td_new.Clone()
    h_ratio.Sumw2()
    h_ratio.Divide(h_td)
    h_ratio.GetYaxis().SetRangeUser(0.0, 2.0)
    h_ratio.SetLineColor(kBlack)
    h_ratio.SetMarkerColor(kBlack)
    h_ratio.SetTitle('')
    h_ratio.GetYaxis().SetNdivisions(205, 1)
    h_ratio.GetXaxis().SetTitle('X Title')
    h_ratio.GetYaxis().SetTitle('data / MC')
    h_ratio.GetXaxis().SetTitleOffset(3)
    h_ratio.Draw('ep')

    c1.Update()
    c1.SaveAs('{}/plot_tuned.{}.{}.{}.pdf'.format(figure_dir, root_filename, root_filename_new, hist_name))
    raw_input('Press any key to continue.')


def plot_minprimdist_scan(**kwargs):
    data_sample = kwargs.get('data_sample', 'fd_cry')
    y_min = kwargs.get('y_min', 0.94)
    y_max = kwargs.get('y_max', 1.0)
    y_axis_title_offset = kwargs.get('y_axis_title_offset')
    plot_purity = kwargs.get('plot_purity', True)

    purities = []
    completenesses = []
    slice_counts = []
    minprimdists = []
    for i in range(4, 9):
        tfile = TFile('{}/{}.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_{}.root'.format(data_dir, data_sample, i))
        h_slicepurity = tfile.Get('tdslicerana/SlicePurity')
        h_slicecompleteness = tfile.Get('tdslicerana/SliceCompleteness')
        h_numslices = tfile.Get('tdslicerana/NumSlices')
        if data_sample == 'fd_genie_nonswap':
            h_slicecompleteness.GetXaxis().SetRangeUser(0.005, 1.005)
            h_slicepurity.GetXaxis().SetRangeUser(0.005, 1.005)

        purities.append(h_slicepurity.GetMean())
        completenesses.append(h_slicecompleteness.GetMean())
        slice_counts.append(h_numslices.GetMean())
        minprimdists.append(float(i))

    gr_purity = TGraph(len(minprimdists), np.array(minprimdists), np.array(purities))
    gr_completeness = TGraph(len(minprimdists), np.array(minprimdists), np.array(completenesses))
    gr_slice_count = TGraph(len(minprimdists), np.array(minprimdists), np.array(slice_counts))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_graph_style(gr_completeness)
    gr_completeness.GetYaxis().SetRangeUser(y_min, y_max)
    gr_completeness.GetYaxis().SetTitle('Completeness{}'.format(' or Purity' if plot_purity else ''))
    gr_completeness.GetXaxis().SetTitle('MinPrimDist')
    if y_axis_title_offset:
        gr_completeness.GetYaxis().SetTitleOffset(y_axis_title_offset)
    gr_completeness.SetLineColor(kRed)
    gr_completeness.SetMarkerColor(gr_completeness.GetLineColor())
    gr_completeness.Draw('ALP')

    if plot_purity:
        set_graph_style(gr_purity)
        gr_purity.SetLineColor(kBlue)
        gr_purity.SetMarkerColor(gr_purity.GetLineColor())
        gr_purity.Draw('LP,sames')

    c1.cd();
    tpad = TPad('tpad', 'tpad', 0, 0, 1, 1)
    tpad.SetFillStyle(4000);
    tpad.SetFrameFillStyle(4000)
    tpad.Draw()
    tpad.cd()
    set_margin()

    set_graph_style(gr_slice_count)
    gr_slice_count.GetYaxis().SetTitle('Slice Count per Event')
    gr_slice_count.GetYaxis().SetTitleOffset(1.0)
    gr_slice_count.Draw('ALP,Y+')

    lg1_y2ndc = 0.4
    if not plot_purity:
        lg1_y2ndc = 0.33
    lg1 = TLegend(0.4, 0.2, 0.6, lg1_y2ndc)
    set_legend_style(lg1)
    lg1.AddEntry(gr_slice_count, 'Slice Count', 'lp')
    lg1.AddEntry(gr_completeness, 'Completeness', 'lp')
    if plot_purity:
        lg1.AddEntry(gr_purity, 'Purity', 'lp')
    lg1.Draw()

    c1.cd()
    c1.Update()
    c1.SaveAs('{}/plot_minprimdist_scan.{}.pdf'.format(figure_dir, data_sample))
    raw_input('Press any key to continue.')


def get_slice_count_completeness_purity(filename, slicer):
    tfile = TFile('{}/{}'.format(data_dir, filename))
    h_slicepurity = tfile.Get('{}/SlicePurity'.format(slicer))
    h_slicecompleteness = tfile.Get('{}/SliceCompleteness'.format(slicer))
    h_numslices = tfile.Get('{}/NumSlices'.format(slicer))
    if 'fd_genie_nonswap' in filename:
        h_slicecompleteness.GetXaxis().SetRangeUser(0.005, 1.005)
        h_slicepurity.GetXaxis().SetRangeUser(0.005, 1.005)
    return h_numslices.GetMean(), h_slicecompleteness.GetMean(), h_slicepurity.GetMean()


def print_slice_count_completeness_purity(data_sample):
    print(data_sample)

    filenames = [
        'ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root',
        'ZScale_27.TScale_57.Tolerance_15.MinPrimDist_4.root',
        'ZScale_27.TScale_57.Tolerance_15.MinPrimDist_5.root',
        'ZScale_27.TScale_57.Tolerance_15.MinPrimDist_6.root',
        'ZScale_27.TScale_57.Tolerance_15.MinPrimDist_7.root',
        'ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root'
    ]
    filenames = map(lambda x: data_sample + '.' + x, filenames)


    if data_sample == 'fd_genie_nonswap':
        slice_count, completeness, purity = get_slice_count_completeness_purity(filenames[4], 'trueslicerana')
        print('TruthSlicer & {:.1f} & {:.3f} \\\\'.format(slice_count, completeness))
        slice_count, completeness, purity = get_slice_count_completeness_purity(filenames[0], 'slicerana')
        print('Slicer4D & {:.1f} & {:.3f} \\\\'.format(slice_count, completeness))
    elif data_sample == 'fd_cry':
        slice_count, completeness, purity = get_slice_count_completeness_purity(filenames[4], 'trueslicerana')
        print('TruthSlicer & {:.1f} & {:.3f} & {:.3f} \\\\'.format(slice_count, completeness, purity))
        slice_count, completeness, purity = get_slice_count_completeness_purity(filenames[0], 'slicerana')
        print('Slicer4D & {:.1f} & {:.3f} & {:.3f} \\\\'.format(slice_count, completeness, purity))

    for i, filename in enumerate(filenames):
        slice_count, completeness, purity = get_slice_count_completeness_purity(filename, 'tdslicerana')
        z_scale = (filename.split('.')[1]).split('_')[1]
        t_scale = (filename.split('.')[2]).split('_')[1]
        tolerance = (filename.split('.')[3]).split('_')[1]
        minprimdist = (filename.split('.')[4]).split('_')[1]
        first_column = '\\texttt{{MinPrimDist}} = {}'.format(minprimdist)
        if i == 0:
            first_column = 'untuned TDSlicer'
        if data_sample == 'fd_genie_nonswap':
            print('{} & {:.1f} & {:.3f} \\\\'.format(first_column, slice_count, completeness))
        elif data_sample == 'fd_cry':
            print('{} & {:.1f} & {:.3f} & {:.3f} \\\\'.format(first_column, slice_count, completeness, purity))
        if i == 0:
            print('\\hline')
            print('\\hline')


def hadd():
    for tolerance in [6, 10]:
        for minprimdist in [4, 5, 6, 7, 8, 9, 10]:
            # call('hadd fd_genie_nonswap_Tolerance_10_MinPrimDist_{}.root data/*fd_genie_nonswap_MinPrimDist_{}.root'.format(i, i), shell=True)
            # call('hadd fd_cry_Tolerance_10_MinPrimDist_{}.root data/*fd_cry_MinPrimDist_{}.root'.format(i, i), shell=True)
            # call('mv fd_genie_nonswap_Tolerance_10_MinPrimDist_{}.root fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_10.MinPrimDist_{}.root'.format(i, i), shell=True)
            # call('mv fd_cry_Tolerance_10_MinPrimDist_{}.root fd_cry.ZScale_27.TScale_57.Tolerance_10.MinPrimDist_{}.root'.format(i, i), shell=True)
            # call('hadd fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_{}.root scratch/*fd_genie_nonswap_MinPrimDist_{}.root'.format(i, i), shell=True)
            # call('hadd fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_{}.root scratch/*fd_cry_MinPrimDist_{}.root'.format(i, i), shell=True)
            call('hadd data/20170927/fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_{}.MinPrimDist_{}.root data/20170927.tmp/*fd_genie_nonswap_minprimdist_{}_tolerance_{}.root'.format(tolerance, minprimdist, minprimdist, tolerance), shell=True)


def get_minprimdist_scan_graphs(data_sample, tolerance):
    purities = []
    completenesses = []
    slice_counts = []
    minprimdists = []
    for i in range(4, 9):
        tfile = TFile('{}/{}.ZScale_27.TScale_57.Tolerance_{}.MinPrimDist_{}.root'.format(data_dir, data_sample, tolerance, i))
        h_slicepurity = tfile.Get('tdslicerana/SlicePurity')
        h_slicecompleteness = tfile.Get('tdslicerana/SliceCompleteness')
        h_numslices = tfile.Get('tdslicerana/NumSlices')
        if data_sample == 'fd_genie_nonswap':
            h_slicecompleteness.GetXaxis().SetRangeUser(0.005, 1.005)
            h_slicepurity.GetXaxis().SetRangeUser(0.005, 1.005)

        purities.append(h_slicepurity.GetMean())
        completenesses.append(h_slicecompleteness.GetMean())
        slice_counts.append(h_numslices.GetMean())
        minprimdists.append(float(i))

    gr_purity = TGraph(len(minprimdists), np.array(minprimdists), np.array(purities))
    gr_completeness = TGraph(len(minprimdists), np.array(minprimdists), np.array(completenesses))
    gr_slice_count = TGraph(len(minprimdists), np.array(minprimdists), np.array(slice_counts))

    return gr_purity, gr_completeness, gr_slice_count


def plot_minprimdist_tolerance_scan(**kwargs):
    data_sample = kwargs.get('data_sample', 'fd_cry')
    y_min = kwargs.get('y_min', 0.94)
    y_max = kwargs.get('y_max', 1.0)
    y_axis_title_offset = kwargs.get('y_axis_title_offset')
    plot_purity = kwargs.get('plot_purity', True)

    gr_purity, gr_completeness, gr_slice_count = get_minprimdist_scan_graphs(data_sample, 10)
    gr_purity_6, gr_completeness_6, gr_slice_count_6 = get_minprimdist_scan_graphs(data_sample, 6)
    gr_purity_15, gr_completeness_15, gr_slice_count_15 = get_minprimdist_scan_graphs(data_sample, 15)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_graph_style(gr_completeness)
    gr_completeness.GetYaxis().SetRangeUser(y_min, y_max)
    gr_completeness.GetYaxis().SetTitle('Completeness{}'.format(' or Purity' if plot_purity else ''))
    gr_completeness.GetXaxis().SetTitle('MinPrimDist')
    if y_axis_title_offset:
        gr_completeness.GetYaxis().SetTitleOffset(y_axis_title_offset)
    gr_completeness.SetLineColor(kRed)
    gr_completeness.SetMarkerColor(gr_completeness.GetLineColor())
    gr_completeness.Draw('AL')

    set_graph_style(gr_completeness_6)
    gr_completeness_6.SetLineStyle(2)
    gr_completeness_6.SetLineColor(gr_completeness.GetLineColor())
    gr_completeness_6.SetLineWidth(gr_completeness.GetLineWidth())
    gr_completeness_6.Draw('L,same')

    set_graph_style(gr_completeness_15)
    gr_completeness_15.SetLineStyle(10)
    gr_completeness_15.SetLineColor(gr_completeness.GetLineColor())
    gr_completeness_15.SetLineWidth(gr_completeness.GetLineWidth())
    gr_completeness_15.Draw('L,same')

    if plot_purity:
        set_graph_style(gr_purity)
        gr_purity.SetLineColor(kBlue)
        gr_purity.SetMarkerColor(gr_purity.GetLineColor())
        gr_purity.Draw('L,sames')

        set_graph_style(gr_purity_6)
        gr_purity_6.SetLineStyle(2)
        gr_purity_6.SetLineColor(gr_purity.GetLineColor())
        gr_purity_6.SetLineWidth(gr_purity.GetLineWidth())
        gr_purity_6.Draw('L,same')

        set_graph_style(gr_purity_15)
        gr_purity_15.SetLineStyle(10)
        gr_purity_15.SetLineColor(gr_purity.GetLineColor())
        gr_purity_15.SetLineWidth(gr_purity.GetLineWidth())
        gr_purity_15.Draw('L,same')

    c1.cd();
    tpad = TPad('tpad', 'tpad', 0, 0, 1, 1)
    tpad.SetFillStyle(4000);
    tpad.SetFrameFillStyle(4000)
    tpad.Draw()
    tpad.cd()
    set_margin()

    set_graph_style(gr_slice_count)
    gr_slice_count.GetYaxis().SetTitle('Slice Count per Event')
    gr_slice_count.GetYaxis().SetTitleOffset(1.0)
    if data_sample == 'fd_cry':
        gr_slice_count.GetYaxis().SetRangeUser(42, 52)
    if data_sample == 'fd_genie_nonswap':
        gr_slice_count.GetYaxis().SetRangeUser(52, 68)
    gr_slice_count.Draw('AL,Y+')

    set_graph_style(gr_slice_count_6)
    gr_slice_count_6.SetLineStyle(2)
    gr_slice_count_6.Draw('L,same')

    set_graph_style(gr_slice_count_15)
    gr_slice_count_15.SetLineStyle(10)
    gr_slice_count_15.Draw('L,same')

    lg1_y2ndc = 0.42
    if not plot_purity:
        lg1_y2ndc = 0.39
    lg1 = TLegend(0.42, 0.31, 0.88, lg1_y2ndc)
    set_legend_style(lg1)
    lg1.SetTextSize(20)
    lg1.SetMargin(0.2)
    lg1.AddEntry(gr_slice_count, 'Slice Count', 'l')
    lg1.AddEntry(gr_completeness, 'Completeness', 'l')
    if plot_purity:
        lg1.AddEntry(gr_purity, 'Purity', 'l')

    lg2 = TLegend(0.42, 0.18, 0.88, 0.29)
    set_legend_style(lg2)
    lg2.SetTextSize(20)
    lg2.SetMargin(0.2)
    lg2.AddEntry(gr_slice_count_6, 'Tolerance = 6', 'l')
    lg2.AddEntry(gr_slice_count, 'Tolerance = 10', 'l')
    lg2.AddEntry(gr_slice_count_15, 'Tolerance = 15', 'l')
    lg2.Draw()
    lg1.Draw()

    c1.cd()
    c1.Update()
    c1.SaveAs('{}/plot_minprimdist_tolerance_scan.{}.pdf'.format(figure_dir, data_sample))
    raw_input('Press any key to continue.')


def get_minprimdist_scan_graphs_genie(data_sample, tolerance):
    purities = []
    completenesses = []
    slice_counts = []
    minprimdists = map(lambda x: float(x), range(4, 11))
    for minprimdist in minprimdists:
        tfile = TFile('{}/{}.ZScale_27.TScale_57.Tolerance_{}.MinPrimDist_{:.0f}.root'.format(data_dir, data_sample, tolerance, minprimdist))
        h_slicepurity = tfile.Get('tdslicerana/fNuPurityByRecoHitGeV')
        h_slicecompleteness = tfile.Get('tdslicerana/fNuCompleteness')
        h_numslices = tfile.Get('tdslicerana/fSliceCountWithNuNueContainment')

        purities.append(h_slicepurity.GetMean())
        completenesses.append(h_slicecompleteness.GetMean())
        print('\\texttt{{MinPrimDist}} = {:.0f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(minprimdist,
                                                                                        h_numslices.GetBinContent(1) / h_numslices.Integral(),
                                                                                        h_numslices.GetBinContent(2) / h_numslices.Integral(),
                                                                                        h_numslices.GetBinContent(3) / h_numslices.Integral()))

        slice_counts.append(h_numslices.GetBinContent(h_numslices.FindBin(1.0)) / h_numslices.Integral())

    gr_purity = TGraph(len(minprimdists), np.array(minprimdists), np.array(purities))
    gr_completeness = TGraph(len(minprimdists), np.array(minprimdists), np.array(completenesses))
    gr_slice_count = TGraph(len(minprimdists), np.array(minprimdists), np.array(slice_counts))

    return gr_purity, gr_completeness, gr_slice_count


def plot_minprimdist_tolerance_scan_genie(**kwargs):
    data_sample = kwargs.get('data_sample', 'fd_cry')
    y_min = kwargs.get('y_min', 0.94)
    y_max = kwargs.get('y_max', 1.0)
    y_axis_title_offset = kwargs.get('y_axis_title_offset')

    gr_purity, gr_completeness, gr_slice_count = get_minprimdist_scan_graphs_genie(data_sample, 10)
    gr_purity_6, gr_completeness_6, gr_slice_count_6 = get_minprimdist_scan_graphs_genie(data_sample, 6)
    gr_purity_15, gr_completeness_15, gr_slice_count_15 = get_minprimdist_scan_graphs_genie(data_sample, 15)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetRightMargin(0.15)
    set_graph_style(gr_completeness)
    gr_completeness.GetYaxis().SetRangeUser(y_min, y_max)
    gr_completeness.GetYaxis().SetTitle('Completeness or Purity')
    gr_completeness.GetXaxis().SetTitle('MinPrimDist')
    if y_axis_title_offset:
        gr_completeness.GetYaxis().SetTitleOffset(y_axis_title_offset)
    gr_completeness.SetLineColor(kRed)
    gr_completeness.SetMarkerColor(gr_completeness.GetLineColor())
    gr_completeness.SetLineStyle(2)
    gr_completeness.Draw('AL')

    set_graph_style(gr_completeness_6)
    gr_completeness_6.SetLineStyle(10)
    gr_completeness_6.SetLineColor(gr_completeness.GetLineColor())
    gr_completeness_6.SetLineWidth(gr_completeness.GetLineWidth())
    gr_completeness_6.Draw('L,same')

    set_graph_style(gr_completeness_15)
    gr_completeness_15.SetLineColor(gr_completeness.GetLineColor())
    gr_completeness_15.SetLineWidth(gr_completeness.GetLineWidth())
    gr_completeness_15.Draw('L,same')


    set_graph_style(gr_purity)
    gr_purity.SetLineColor(kBlue)
    gr_purity.SetMarkerColor(gr_purity.GetLineColor())
    gr_purity.SetLineStyle(2)
    gr_purity.Draw('L,sames')

    set_graph_style(gr_purity_6)
    gr_purity_6.SetLineStyle(10)
    gr_purity_6.SetLineColor(gr_purity.GetLineColor())
    gr_purity_6.SetLineWidth(gr_purity.GetLineWidth())
    gr_purity_6.Draw('L,same')

    set_graph_style(gr_purity_15)
    gr_purity_15.SetLineColor(gr_purity.GetLineColor())
    gr_purity_15.SetLineWidth(gr_purity.GetLineWidth())
    gr_purity_15.Draw('L,same')

    c1.cd();
    tpad = TPad('tpad', 'tpad', 0, 0, 1, 1)
    tpad.SetFillStyle(4000);
    tpad.SetFrameFillStyle(4000)
    tpad.Draw()
    tpad.cd()
    set_margin()
    gPad.SetRightMargin(0.15)

    set_graph_style(gr_slice_count)
    gr_slice_count.GetYaxis().SetTitle('Fraction of Events with One #nu-Slice ')
    gr_slice_count.GetYaxis().SetTitleOffset(1.3)
    gr_slice_count.GetYaxis().SetRangeUser(0.3, 0.6)
    gr_slice_count.SetLineStyle(2)
    gr_slice_count.Draw('AL,Y+')

    set_graph_style(gr_slice_count_6)
    gr_slice_count_6.SetLineStyle(10)
    gr_slice_count_6.Draw('L,same')

    set_graph_style(gr_slice_count_15)
    gr_slice_count_15.Draw('L,same')

    lg1 = TLegend(0.26, 0.3, 0.72, 0.41)
    set_legend_style(lg1)
    lg1.SetTextSize(20)
    lg1.SetMargin(0.2)
    lg1.AddEntry(gr_slice_count_15, 'Event fraction with one #nu-slice', 'l')
    lg1.AddEntry(gr_completeness_15, 'Completeness', 'l')
    lg1.AddEntry(gr_purity_15, 'Purity', 'l')

    lg2 = TLegend(0.26, 0.18, 0.72, 0.29)
    set_legend_style(lg2)
    lg2.SetTextSize(20)
    lg2.SetMargin(0.2)
    lg2.AddEntry(gr_slice_count_6, 'Tolerance = 6', 'l')
    lg2.AddEntry(gr_slice_count, 'Tolerance = 10', 'l')
    lg2.AddEntry(gr_slice_count_15, 'Tolerance = 15', 'l')
    lg2.Draw()
    lg1.Draw()

    c1.cd()
    c1.Update()
    c1.SaveAs('{}/plot_minprimdist_tolerance_scan_genie.{}.pdf'.format(figure_dir, data_sample))
    raw_input('Press any key to continue.')


def plot_true_nu_count(**kwargs):
    # hist_name = kwargs.get('hist_name', 'NumSlices')
    # x_min = kwargs.get('x_min')
    # x_max = kwargs.get('x_max')
    # log_y = kwargs.get('log_y', False)
    # log_x = kwargs.get('log_x', False)
    # x_title = kwargs.get('x_title')
    # y_title = kwargs.get('y_title')
    # statbox_position = kwargs.get('statbox_position', 'right')

    hist_name = 'NumNu'
    root_filename = kwargs.get('root_filename')

    f_slicer = TFile('{}/{}'.format(data_dir, root_filename))
    h_4d = f_slicer.Get('slicerana/{}'.format(hist_name))
    # h_td = f_slicer.Get('tdslicerana/{}'.format(hist_name))

    c1 = TCanvas('c1', 'c1', 800, 600)
    gStyle.SetOptStat('emr')
    set_margin()
    gPad.SetLogy()
    # gPad.SetLogy(log_y)
    # gPad.SetLogx(log_x)

    set_h1_style(h_4d)
    # h_4d.SetName('slicer4d')
    h_4d.SetName('')
    # if x_min and x_max:
    h_4d.GetXaxis().SetRangeUser(0, 2)
    # if not log_y:
    #     h_4d.GetYaxis().SetRangeUser(0, get_max_y([h_4d, h_td]) * 1.1)
    # if x_title:
    #     h_4d.GetXaxis().SetTitle(x_title)
    # if y_title:
    #     h_4d.GetYaxis().SetTitle(y_title)
    h_4d.Draw()

    # set_h1_style(h_td)
    # h_td.SetName('tdslicer')
    # h_td.SetLineColor(kRed)
    # h_td.Draw('sames')

    c1.Update()
    draw_statbox(h_4d)
    # draw_statboxes(h_td, h_4d, position=statbox_position)

    c1.Update()
    c1.SaveAs('{}/plot_true_nu_count.{}.pdf'.format(figure_dir, root_filename))
    raw_input('Press any key to continue.')


def get_slice_count_nu_no_nu_genie(data_sample, tolerance):
    slice_nu_counts = []
    slice_nonu_counts = []
    slice_ratios = []
    minprimdists = map(lambda x: float(x), range(4, 11))
    for minprimdist in minprimdists:
        tfile = TFile('{}/{}.ZScale_27.TScale_57.Tolerance_{}.MinPrimDist_{:.0f}.root'.format(data_dir, data_sample, tolerance, minprimdist))
        h_nu = tfile.Get('tdslicerana/fSliceCountWithNuNueContainment')
        h_nonu = tfile.Get('tdslicerana/fSliceCountNoNuNueContainment')
        slice_nu_count = h_nu.GetBinContent(h_nu.FindBin(1.0)) / h_nu.Integral()
        slice_nonu_count = h_nonu.GetMean()
        slice_ratio = slice_nu_count / slice_nonu_count
        slice_nu_counts.append(slice_nu_count)
        slice_nonu_counts.append(slice_nonu_count)
        slice_ratios.append(slice_ratio)

    gr_slice_nu_count = TGraph(len(minprimdists), np.array(minprimdists), np.array(slice_nu_counts))
    gr_slice_nonu_count = TGraph(len(minprimdists), np.array(minprimdists), np.array(slice_nonu_counts))
    gr_slice_ratio = TGraph(len(minprimdists), np.array(minprimdists), np.array(slice_ratios))

    return gr_slice_nu_count, gr_slice_nonu_count, gr_slice_ratio


def plot_slice_count_nu_no_nu_genie(**kwargs):
    data_sample = kwargs.get('data_sample', 'fd_cry')
    gr_slice_nu_count, gr_slice_nonu_count, gr_slice_ratio = get_slice_count_nu_no_nu_genie(data_sample, 15)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetRightMargin(0.15)
    set_graph_style(gr_slice_nu_count)
    gr_slice_nu_count.GetYaxis().SetTitle('Fraction of Events with One #nu-slice')
    gr_slice_nu_count.GetXaxis().SetTitle('MinPrimDist')
    gr_slice_nu_count.GetYaxis().SetTitleOffset(1.3)
    gr_slice_nu_count.SetLineColor(kRed)
    gr_slice_nu_count.SetMarkerColor(gr_slice_nu_count.GetLineColor())
    gr_slice_nu_count.Draw('AL')

    c1.cd();
    tpad = TPad('tpad', 'tpad', 0, 0, 1, 1)
    tpad.SetFillStyle(4000);
    tpad.SetFrameFillStyle(4000)
    tpad.Draw()
    tpad.cd()
    set_margin()
    gPad.SetRightMargin(0.15)
    set_graph_style(gr_slice_nonu_count)
    gr_slice_nonu_count.GetYaxis().SetTitle('Average Cosmic-slice Count')
    gr_slice_nonu_count.GetYaxis().SetTitleOffset(1.3)
    gr_slice_nonu_count.Draw('AL,Y+')
    lg1 = TLegend(0.18, 0.18, 0.64, 0.28)
    set_legend_style(lg1)
    lg1.SetTextSize(20)
    lg1.SetMargin(0.2)
    lg1.AddEntry(gr_slice_nu_count, 'fraction of events with one #nu-slice', 'l')
    lg1.AddEntry(gr_slice_nonu_count, 'average cosmic-slice count', 'l')
    lg1.Draw()
    c1.cd()
    c1.Update()
    c1.SaveAs('{}/plot_slice_count_nu_no_nu_genie.{}.pdf'.format(figure_dir, data_sample))

    c2 = TCanvas('c2', 'c2', 800, 600)
    set_margin()
    gPad.SetRightMargin(0.15)
    gPad.SetGrid()
    set_graph_style(gr_slice_ratio)
    gr_slice_ratio.GetYaxis().SetTitle('Ratio of signal to noise')
    gr_slice_ratio.GetXaxis().SetTitle('MinPrimDist')
    gr_slice_ratio.GetYaxis().SetTitleOffset(1.3)
    gr_slice_ratio.SetMarkerColor(gr_slice_ratio.GetLineColor())
    gr_slice_ratio.Draw('AL')
    c2.cd()
    c2.Update()
    c2.SaveAs('{}/plot_slice_count_nu_no_nu_genie.ratio.{}.pdf'.format(figure_dir, data_sample))

    raw_input('Press any key to continue.')

def print_slicer4d_vs_tdslicer_genie():
    data_sample = 'fd_genie_nonswap'

    tfile_untuned = TFile('{}/{}.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root'.format(data_dir, data_sample))
    h_slicepurity_td_untuned = tfile_untuned.Get('tdslicerana/fNuPurityByRecoHitGeV')
    h_slicecompleteness_td_untuned = tfile_untuned.Get('tdslicerana/fNuCompleteness')
    h_slicecount_withnu_td_untuned = tfile_untuned.Get('tdslicerana/fSliceCountWithNuNueContainment')
    h_slicecount_nonu_td_untuned = tfile_untuned.Get('tdslicerana/fSliceCountNoNuNueContainment')

    tfile = TFile('{}/{}.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root'.format(data_dir, data_sample))
    h_slicepurity_td = tfile.Get('tdslicerana/fNuPurityByRecoHitGeV')
    h_slicecompleteness_td = tfile.Get('tdslicerana/fNuCompleteness')
    h_slicecount_withnu_td = tfile.Get('tdslicerana/fSliceCountWithNuNueContainment')
    h_slicecount_nonu_td = tfile.Get('tdslicerana/fSliceCountNoNuNueContainment')

    h_slicepurity_4d = tfile.Get('slicerana/fNuPurityByRecoHitGeV')
    h_slicecompleteness_4d = tfile.Get('slicerana/fNuCompleteness')
    h_slicecount_withnu_4d = tfile.Get('slicerana/fSliceCountWithNuNueContainment')
    h_slicecount_nonu_4d = tfile.Get('slicerana/fSliceCountNoNuNueContainment')

    signal_4d = h_slicecount_withnu_4d.GetBinContent(h_slicecount_withnu_4d.FindBin(1.0)) / h_slicecount_withnu_4d.Integral()
    signal_td = h_slicecount_withnu_td.GetBinContent(h_slicecount_withnu_td.FindBin(1.0)) / h_slicecount_withnu_td.Integral()
    signal_td_untuned = h_slicecount_withnu_td_untuned.GetBinContent(h_slicecount_withnu_td_untuned.FindBin(1.0)) / h_slicecount_withnu_td_untuned.Integral()

    noise_4d = h_slicecount_nonu_4d.GetMean()
    noise_td = h_slicecount_nonu_td.GetMean()
    noise_td_untuned = h_slicecount_nonu_td_untuned.GetMean()

    print('parameter & Slicer4D & untuned TDSlicer & tuned TDSlicer \\\\')
    print('\\hline')
    print('\\hline')
    print('purity & {:.3f} & {:.3f} & {:.3f} \\\\'.format(h_slicepurity_4d.GetMean(), h_slicepurity_td_untuned.GetMean(), h_slicepurity_td.GetMean()))
    print('completeness & {:.3f} & {:.3f} & {:.3f} \\\\'.format(h_slicecompleteness_4d.GetMean(), h_slicecompleteness_td_untuned.GetMean(), h_slicecompleteness_td.GetMean()))
    print('\\hline')
    print('\\hline')
    print('signal strength & {:.3f} & {:.3f} & {:.3f} \\\\'.format(signal_4d, signal_td_untuned, signal_td))
    print('noise strength & {:.3f} & {:.3f} & {:.3f} \\\\'.format(noise_4d, noise_td_untuned, noise_td))
    print('signal-to-noise ratio & {:.3f} & {:.3f} & {:.3f}'.format(signal_4d / noise_4d, signal_td_untuned / noise_td_untuned, signal_td / noise_td))


def plot_fls_hit(filename):
    f_slicer = TFile('{}/{}'.format(data_dir, filename))
    h_fls_hit_count = f_slicer.Get('slicerana/fFlsHitCount')
    h_fls_hit_gev = f_slicer.Get('slicerana/fFlsHitGeV')

    gStyle.SetOptStat('emr')
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetRightMargin(0.15)
    set_h1_style(h_fls_hit_count)
    h_fls_hit_count.Rebin(5)
    h_fls_hit_count.GetXaxis().SetTitle('FLS Hit Count')
    h_fls_hit_count.GetYaxis().SetTitle('Event Count')
    # h_fls_hit_count.GetXaxis().SetRangeUser(0, 500)
    h_fls_hit_count.Draw()
    c1.Update()
    draw_statbox(h_fls_hit_count)
    c1.Update()
    c1.SaveAs('{}/plot_fls_hit.h_fls_hit_count.pdf'.format(figure_dir))

    # c2 = TCanvas('c2', 'c2', 800, 600)
    # set_margin()
    # gPad.SetRightMargin(0.15)
    # set_h1_style(h_fls_hit_gev)
    # h_fls_hit_gev.Rebin(5)
    # h_fls_hit_gev.GetXaxis().SetTitle('FLS Hit Energy Deposition (Gev)')
    # h_fls_hit_gev.GetYaxis().SetTitle('Event Count')
    # h_fls_hit_gev.GetXaxis().SetRangeUser(0, 500)
    # h_fls_hit_gev.Draw('colz')
    # c2.Update()
    # draw_statbox(h_fls_hit_gev)
    # c2.Update()
    # c2.SaveAs('{}/plot_fls_hit.h_fls_hit_gev.pdf'.format(figure_dir))

    input('Press any key to continue.')


def plot_fls_hit_xy(filename):
    f_slicer = TFile('{}/{}'.format(data_dir, filename))
    h_fls_hit_count_xy = f_slicer.Get('slicerana/fFlsHitCountXY')
    h_fls_hit_gev_xy = f_slicer.Get('slicerana/fFlsHitGeVXY')

    print('h_fls_hit_count_xy.GetEntries() = ', h_fls_hit_count_xy.GetEntries())
    print('h_fls_hit_count_xy.Integral(1, 4, 1, 4) = ', h_fls_hit_count_xy.Integral(1, 4, 1, 4))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gStyle.SetOptStat(0)
    gPad.SetRightMargin(0.15)
    gPad.SetLogz()

    set_h2_color_style()
    set_h2_style(h_fls_hit_count_xy)
    # h_fls_hit_count_xy.Rebin2D(2, 2)
    h_fls_hit_count_xy.GetXaxis().SetTitle('FLS Hit Count in X View')
    h_fls_hit_count_xy.GetYaxis().SetTitle('FLS Hit Count in Y View')
    h_fls_hit_count_xy.GetXaxis().SetRangeUser(0, 20)
    h_fls_hit_count_xy.GetYaxis().SetRangeUser(0, 20)
    h_fls_hit_count_xy.Draw('colz')
    c1.Update()
    c1.SaveAs('{}/plot_fls_hit_xy.h_fls_hit_count_xy.pdf'.format(figure_dir))

    # c2 = TCanvas('c2', 'c2', 800, 600)
    # set_margin()
    # gStyle.SetOptStat(0)
    # gPad.SetRightMargin(0.15)
    # set_h2_style(h_fls_hit_gev_xy)
    # h_fls_hit_gev_xy.Rebin2D(5, 5)
    # h_fls_hit_gev_xy.GetXaxis().SetTitle('FLS Hit Energy Deposition in X View (GeV)')
    # h_fls_hit_gev_xy.GetYaxis().SetTitle('FLS Hit Energy Deposition in Y View (GeV)')
    # h_fls_hit_gev_xy.GetXaxis().SetRangeUser(0, 3)
    # h_fls_hit_gev_xy.GetYaxis().SetRangeUser(0, 3)
    # h_fls_hit_gev_xy.Draw('colz')
    # c2.Update()
    # c2.SaveAs('{}/plot_fls_hit_xy.h_fls_hit_gev_xy.pdf'.format(figure_dir))

    input('Press any key to continue.')


# gStyle.SetOptStat('emr')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.FLS.root', hist_name='fSliceCountWithNuNueContainment0GeV', statbox_position='right', x_min=-0.1, x_max=5, x_title='Total FLS Hit Energy Deposition (GeV)', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.FLS.root', hist_name='fSliceCountWithNuNueContainment1GeV', statbox_position='right', x_min=-0.1, x_max=5, x_title='Total FLS Hit Energy Deposition (GeV)', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.FLS.root', hist_name='fSliceCountWithNuNueContainment', statbox_position='right', x_min=-0.5, x_max=3, x_title='Number of Slices With Contributions from #nu', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.FLS.root', hist_name='fSliceCountWithNuNumuContainment', statbox_position='right', x_min=-0.5, x_max=3, x_title='Number of Slices With Contributions from #nu', y_title='Event Count')
# plot_fls_hit('fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.FLS.root')
# plot_fls_hit_xy('fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.FLS.root')


# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='SlicePurity', log_y=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='SliceCompleteness', log_y=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='NumSlices', log_y=False, statbox_position='right', x_min=20, x_max=100)

# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='NumSlices', log_y=False, statbox_position='left', x_min=10, x_max=120)
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='SliceCompleteness', log_y=True, statbox_position='top')

# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='SlicePurity', log_y=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='SliceCompleteness', log_y=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root',
#      hist_name='NumSlices', log_y=False, statbox_position='right', x_min=20, x_max=100)

# print_slice_count_completeness_purity('fd_genie_nonswap')
# print_slice_count_completeness_purity('fd_cry')
# plot_minprimdist_scan(data_sample='fd_cry')
# plot_minprimdist_tolerance_scan(data_sample='fd_cry')
# plot_minprimdist_tolerance_scan(data_sample='fd_genie_nonswap', plot_purity=False)
# hadd()

# print_slice_count_completeness_purity('fd_genie_nonswap')
# print_slice_count_completeness_purity('fd_cry')

# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#      hist_name='SliceCompleteness', log_y=True, statbox_left=True, statbox_position='top')

# plot_minprimdist_scan(data_sample='fd_cry')
# plot_minprimdist_scan(data_sample='fd_genie_nonswap', plot_purity=False)

# plot(hist_name='NumSlices', x_min=30, x_max=200)
# plot(hist_name='NumMCT', x_min=90, x_max=200, statbox_position='left')
# plot(hist_name='NumSliceHits', x_min=-1, x_max=500)
# plot(hist_name='SliceCompleteness', log_y=True, statbox_left=True, statbox_position='top')
# plot(hist_name='SlicePurity', log_y=True, statbox_left=True, statbox_position='left')
# plot(hist_name='NumNoiseSlices', x_min=-0.5, x_max=3, log_y=True)

# plot_slice_matching('fSliceTree')
# plot_slice_matching('fNuSliceTree')

# plot_tuned(hist_name='NumSlices', x_min=30, x_max=200)
# plot_tuned(hist_name='NumSlices', x_min=20, x_max=100, root_filename='fd_cry.root', root_filename_new='fd_cry.new.root')

# plot_tuned(hist_name='NumSlices',
#            root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#            root_filename_new='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_10.MinPrimDist_6.root')
# plot_tuned(hist_name='NumSlices',
#            root_filename='fd_genie.nonswap.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_genie.nonswap.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_6.root')
# plot_tuned(hist_name='NumSlices',
#            root_filename='fd_genie.nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root')

# plot(hist_name='SliceCompleteness', log_y=True, statbox_left=True, statbox_position='top')
# plot(hist_name='SlicePurity', log_y=True, statbox_left=True, statbox_position='left')

# plot_ratio(hist_name='SliceCompleteness',
#            root_filename='fd_genie.nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#            log_y=True)
# plot_ratio(hist_name='SlicePurity',
#            root_filename='fd_genie.nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#            log_y=True)

# plot_tuned(hist_name='NumSlices',
#            root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root')
# plot_ratio(hist_name='SliceCompleteness',
#            root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#            log_y=True)
# plot_ratio(hist_name='SlicePurity',
#            root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#            root_filename_new='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#            log_y=True)
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#      hist_name='SliceCompleteness', log_y=True, statbox_left=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#      hist_name='SliceCompleteness', log_y=True, statbox_left=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_4.root',
#      hist_name='SlicePurity', log_y=True, statbox_left=True, statbox_position='top')
# plot(root_filename='fd_cry.ZScale_27.TScale_57.Tolerance_6.MinPrimDist_6.root',
#      hist_name='SlicePurity', log_y=True, statbox_left=True, statbox_position='top')

# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fNuPurityByRecoHitGeV', log_y=True, statbox_position='top')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fNuPurityByRecoHitCount', log_y=True, statbox_position='top')

# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fSliceCountWithNu', statbox_position='right', x_min=-0.5, x_max=3, x_title='Number of Slices With Contributions from #nu', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fSliceCountNoNu', statbox_position='left', x_min=9.5, x_max=100.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')

# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fSliceCountWithNuNueContainment', statbox_position='right', x_min=-0.5, x_max=3, x_title='Number of Slices With Contributions from #nu', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fSliceCountNoNuNueContainment', statbox_position='right', x_min=-0.5, x_max=15.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')

# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fSliceCountWithNuNumuContainment', statbox_position='right', x_min=-0.5, x_max=3, x_title='Number of Slices With Contributions from #nu', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root', hist_name='fSliceCountNoNuNumuContainment', statbox_position='right', x_min=-0.5, x_max=15.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')

# hadd()
# plot_minprimdist_tolerance_scan_genie(data_sample='fd_genie_nonswap', y_axis_title_offset=1.3)
# plot_true_nu_count(root_filename='fd_genie_nonswap.ZScale_27.TScale_57.Tolerance_15.MinPrimDist_8.root')
# plot_slice_count_nu_no_nu_genie(data_sample='fd_genie_nonswap')
# print_slicer4d_vs_tdslicer_genie()

# plot(root_filename='fd_genie_nonswap.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root', hist_name='fSliceCountNoNu', statbox_position='left', x_min=-0.5, x_max=100.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root', hist_name='fSliceCountNoNuNueContainment', statbox_position='right', x_min=-0.5, x_max=20.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')

# plot(root_filename='fd_genie_nonswap.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root', hist_name='fSliceCountWithNu', statbox_position='right', x_min=-0.5, x_max=3.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')
# plot(root_filename='fd_genie_nonswap.ZScale_100.TScale_10.Tolerance_6.MinPrimDist_4.root', hist_name='fSliceCountWithNuNueContainment', statbox_position='right', x_min=-0.5, x_max=3.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')
# plot(root_filename='SlicerAna_hist.containment.root', hist_name='fSliceCountWithNuNumuContainment', statbox_position='right', x_min=-0.5, x_max=3, x_title='Number of Slices With Contributions from #nu', y_title='Event Count')
# plot(root_filename='SlicerAna_hist.containment.root', hist_name='fSliceCountNoNuNumuContainment', statbox_position='right', x_min=-0.5, x_max=15.5, x_title='Number of Slices With No Contribution from #nu', y_title='Event Count')

# 20171022_tdslicer_nd_genie
plot(root_filename='nd_genie.root', hist_name='SlicePurity', statbox_position='right')
