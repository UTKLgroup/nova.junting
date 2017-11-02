from rootalias import *

figure_dir = '/Users/juntinghuang/beamer/20171030_data_new_physics/figures'

top_ys = [755., 225.]
top_zs = [1826., 2779.]
bottom_ys = [181., -343.]
bottom_zs = [2586., 3438.]

gr_top = TGraph(len(top_ys), np.array(top_zs), np.array(top_ys))
gr_bottom = TGraph(len(bottom_ys), np.array(bottom_zs), np.array(bottom_ys))

c1 = TCanvas('c1', 'c1', 800, 600)
set_margin()

gr_top.GetYaxis().SetRangeUser(-500, 800)
gr_top.GetXaxis().SetLimits(1500, 3500)
gr_top.SetMarkerSize(1.5)
gr_top.SetMarkerStyle(20)
gr_top.Draw('AP')
gr_top.Fit('pol1')
f_top = gr_top.GetFunction('pol1')
b_top = f_top.GetParameter(0)
k_top = f_top.GetParameter(1)

gr_bottom.SetMarkerSize(1.5)
gr_bottom.SetMarkerStyle(20)
gr_bottom.Draw('sames,P')
gr_bottom.Fit('pol1')
f_bottom = gr_bottom.GetFunction('pol1')
b_bottom = f_bottom.GetParameter(0)
k_bottom = f_bottom.GetParameter(1)

print(k_top, b_top)
print(k_bottom, b_bottom)

z0 = (b_bottom - b_top) / (k_top - k_bottom)
y0 = k_top * z0 + b_top
print(z0, y0)

c1.Update()
c1.SaveAs('{}/vertex.pdf'.format(figure_dir))
input('Press any key to continue.')
