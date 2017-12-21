from rootalias import *

data_dir = './data'
figure_dir = './figures'

tf = TFile('{}/novelty_hist.root'.format(data_dir))
h_x = TH2D('h1', 'h1', 896, 0, 896, 384, 0, 384)
h_y = TH2D('h1', 'h1', 896, 0, 896, 384, 0, 384)
last_event = None
for sl in tf.Get('noveltyana/fSliceTree'):
    print(last_event, sl.event)
    if last_event and sl.event != last_event:
        print(sl.run, sl.subrun, sl.event)
        break
    last_event = sl.event

    cell_hit_count = len(sl.views)
    for i in range(cell_hit_count):
        cell = sl.cells[i]
        plane = sl.planes[i]
        view = sl.views[i]
        if view == 0:
            h_x.Fill(plane, cell)
        else:
            h_y.Fill(plane, cell)


c1 = TCanvas('c1', 'c1', 800, 600)
gStyle.SetOptStat(0)
set_margin()
set_h2_color_style()
c1.Divide(1, 2)

c1.cd(1)
h_x.Draw('colz')

c1.cd(2)
h_y.Draw('colz')

c1.Update()
c1.SaveAs('{}/novelty.pdf'.format(figure_dir))
input('Press any key to continue.')
