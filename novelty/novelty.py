from rootalias import *
import numpy as np


data_dir = './data'
figure_dir = '/Users/juntinghuang/beamer/20171222_feb_flasher/figures'

filename = 'novelty_hist.pid_grl.hit_count_10.plane_10.containment_200.root'
# filename = 'fd_cry.root'


def plot_feb_flasher():
    tf = TFile('{}/{}'.format(data_dir, filename))

    x_gr_planes = []
    x_gr_cells = []
    y_gr_planes = []
    y_gr_cells = []

    for sl in tf.Get('noveltyana/fSliceTree'):
        cell_hit_count = len(sl.views)
        planes = []
        x_cells = []
        y_cells = []
        for i in range(cell_hit_count):
            planes.append(sl.planes[i])
            if sl.views[i] == 0:
                x_cells.append(sl.cells[i])
            else:
                y_cells.append(sl.cells[i])
        y_cell_extent = max(y_cells) - min(y_cells)
        x_cell_extent = max(x_cells) - min(x_cells)
        x_cell_count = len(x_cells)
        y_cell_count = len(y_cells)
        xy_cell_count = x_cell_count + y_cell_count

        if xy_cell_count < 180:
            continue
        if y_cell_extent > 150:
            continue
        if x_cell_extent > 150:
            continue

        print(sl.run, sl.subrun, sl.event)
        for i in range(cell_hit_count):
            cell = float(sl.cells[i])
            plane = float(sl.planes[i])
            view = sl.views[i]
            if view == 0:
                x_gr_planes.append(plane)
                x_gr_cells.append(cell)
            else:
                y_gr_planes.append(plane)
                y_gr_cells.append(cell)

    c1 = TCanvas('c1', 'c1', 800, 600)
    gStyle.SetOptStat(0)
    set_margin()
    gr_x = TGraph(len(x_gr_cells), np.array(x_gr_planes), np.array(x_gr_cells))
    set_graph_style(gr_x)
    gr_x.GetXaxis().SetTitle('Plane Number')
    gr_x.GetYaxis().SetTitle('Cell Number')
    gr_x.GetXaxis().SetRangeUser(0, 896)
    gr_x.GetYaxis().SetRangeUser(0, 384)
    gr_x.SetMarkerStyle(21)
    gr_x.SetMarkerSize(0.2)
    gr_x.Draw('AP')
    c1.Update()
    c1.SaveAs('{}/plot_feb_flasher.{}.x_view.pdf'.format(figure_dir, filename))

    c2 = TCanvas('c2', 'c2', 800, 600)
    gStyle.SetOptStat(0)
    set_margin()
    gr_y = TGraph(len(y_gr_cells), np.array(y_gr_planes), np.array(y_gr_cells))
    set_graph_style(gr_y)
    gr_y.GetXaxis().SetTitle('Plane Number')
    gr_y.GetYaxis().SetTitle('Cell Number')
    gr_y.GetXaxis().SetRangeUser(0, 896)
    gr_y.GetYaxis().SetRangeUser(0, 384)
    gr_y.SetMarkerStyle(21)
    gr_y.SetMarkerSize(0.2)
    gr_y.Draw('AP')
    c2.Update()
    c2.SaveAs('{}/plot_feb_flasher.{}.y_view.pdf'.format(figure_dir, filename))
    input('Press any key to continue.')


def plot_no_filter():
    tf = TFile('{}/{}'.format(data_dir, filename))

    x_gr_planes = []
    x_gr_cells = []
    y_gr_planes = []
    y_gr_cells = []

    h_x = TH2D('h1', 'h1', 896, 0, 896, 384, 0, 384)
    h_y = TH2D('h1', 'h1', 896, 0, 896, 384, 0, 384)

    event_slice_count = {}

    for sl in tf.Get('noveltyana/fSliceTree'):
        event = sl.event
        if event not in event_slice_count:
            event_slice_count[event] = 1
        else:
            event_slice_count[event] += 1

        cell_hit_count = len(sl.views)
        for i in range(cell_hit_count):
            cell = float(sl.cells[i])
            plane = float(sl.planes[i])
            view = sl.views[i]
            if view == 0:
                h_x.Fill(plane, cell)
            else:
                h_y.Fill(plane, cell)


    h_slice_count = TH1D('h_slice_count', 'h_slice_count', 101, -0.5, 100.5)
    for event in event_slice_count:
        h_slice_count.Fill(event_slice_count[event])

    gStyle.SetOptStat('emr')
    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    set_h1_style(h_slice_count)
    h_slice_count.GetXaxis().SetRangeUser(-0.5, 6.5)
    h_slice_count.GetXaxis().SetTitle('Slice Count')
    h_slice_count.GetYaxis().SetTitle('Event Count')
    h_slice_count.GetYaxis().SetTitleOffset(2)
    h_slice_count.Draw()
    c1.Update()
    draw_statbox(h_slice_count, x1=0.65)

    c1.Update()
    c1.SaveAs('{}/plot_no_filter.h_slice_count.pdf'.format(figure_dir))
    input('Press any key to continue.')


    # c1 = TCanvas('c1', 'c1', 800, 800)
    # gStyle.SetOptStat(0)
    # set_margin()
    # set_h2_color_style()
    # set_h2_style(h_x)
    # h_x.GetXaxis().SetTitle('Plane Number')
    # h_x.GetYaxis().SetTitle('Cell Number')
    # h_x.GetYaxis().SetTitleOffset(1.5)
    # h_x.Draw('colz')
    # c1.Update()
    # c1.SaveAs('{}/plot_no_filter.{}.x_view.png'.format(figure_dir, filename))

    # c2 = TCanvas('c2', 'c2', 800, 800)
    # gStyle.SetOptStat(0)
    # set_margin()
    # set_h2_color_style()
    # set_h2_style(h_y)
    # h_y.GetXaxis().SetTitle('Plane Number')
    # h_y.GetYaxis().SetTitle('Cell Number')
    # h_y.GetYaxis().SetTitleOffset(1.5)
    # h_y.Draw('colz')
    # c2.Update()
    # c2.SaveAs('{}/plot_no_filter.{}.y_view.png'.format(figure_dir, filename))

    input('Press any key to continue.')


# plot_feb_flasher()
plot_no_filter()
