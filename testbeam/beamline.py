import csv
from rootalias import *


class Detector:
    def __init__(self, name):
        # all length in mm
        self.name = name
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.length = 10.
        self.width = 10.
        self.height = 10.
        self.aperture_width = 1.
        self.aperture_height = 1.
        self.color = kBlack
        self.marker_style = 20

    def set_zx(self, zx):
        self.z = zx[0]
        self.x = zx[1]


class Beamline:

    def __init__(self):
        self.figure_dir = None

        self.target = Detector('target')
        self.collimator_us = Detector('upstream collimator')
        self.tof_us = Detector('upstream TOF')
        self.wc_1 = Detector('wire chamber 1')
        self.wc_2 = Detector('wire chamber 2')
        self.magnet = Detector('magnet')
        self.wc_3 = Detector('wire chamber 3')
        self.collimator_ds = Detector('downstream collimator')
        self.wc_4 = Detector('wire chamber 4')
        self.cherenkov = Detector('cherenkov counter')
        self.tof_ds = Detector('downstream TOF')
        self.nova = Detector('nova detector')
        
        self.detectors = [
            self.target,
            self.collimator_us,
            self.tof_us,
            self.wc_1,
            self.wc_2,
            self.magnet,
            self.wc_3,
            self.collimator_ds,
            self.wc_4,
            self.cherenkov,
            self.tof_ds,
            self.nova
        ]

        self.get_position()
        self.get_nova_dimension()
        self.get_magnet_dimension()
        self.get_collimator_us_dimension()
        self.get_collimator_ds_dimension()

    @staticmethod
    def get_average(points):
        dimension_count = len(points[0])
        point_count = len(points)
        average = []
        for i in range(dimension_count):
            average.append(sum([point[i] for point in points]) / point_count)
        return average

    @staticmethod
    def get_distance(point_1, point_2):
        dimension_count = len(point_1)
        distance = 0.
        for i in range(dimension_count):
            distance += (point_1[i] - point_2[i])**2
        return distance**0.5

    @staticmethod
    def get_csv(filename):
        rows = []
        with open(filename) as f_csv:
            csv_reader = csv.reader(f_csv, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                rows.append(list(map(lambda x: float(x) * 10., row)))
        return rows

    def get_position(self):
        rows = Beamline.get_csv('digitize/ftbf_drawing_digitize.csv')
        origin = rows[0]
        rows = [[row[0] - origin[0], row[1] - origin[1]] for row in rows]

        self.target.set_zx(rows[0])
        self.tof_us.set_zx(rows[1])
        self.wc_1.set_zx(Beamline.get_average([rows[2], rows[3]]))
        self.wc_2.set_zx(Beamline.get_average([rows[4], rows[5]]))
        self.magnet.set_zx(Beamline.get_average([rows[6], rows[9]]))
        self.wc_3.set_zx(Beamline.get_average([rows[10], rows[11]]))
        self.collimator_ds.set_zx(Beamline.get_average([rows[12], rows[13]]))
        self.wc_4.set_zx(Beamline.get_average([rows[14], rows[15]]))
        self.cherenkov.set_zx(Beamline.get_average([rows[16], rows[17]]))
        self.tof_ds.set_zx(rows[18])
        self.nova.set_zx(rows[19])

        collimator_us_points = Beamline.get_csv('digitize/collimator_us.csv')
        collimator_us_position = Beamline.get_average([collimator_us_points[0], collimator_us_points[2], collimator_us_points[3], collimator_us_points[5]])
        self.collimator_us.set_zx([collimator_us_position[0] - origin[0], collimator_us_position[1] - origin[1]])

    def get_nova_dimension(self):
        top_points = Beamline.get_csv('digitize/nova.csv')
        self.nova.length = np.average([Beamline.get_distance(top_points[0], top_points[1]), Beamline.get_distance(top_points[2], top_points[3])])
        self.nova.width = Beamline.get_distance(top_points[1], top_points[2])
        self.nova.height = self.nova.width

    def get_magnet_dimension(self):
        top_points = Beamline.get_csv('digitize/magnet.csv')
        self.magnet.length = np.average([Beamline.get_distance(top_points[0], top_points[1]), Beamline.get_distance(top_points[2], top_points[3])])
        self.magnet.width = Beamline.get_distance(top_points[1], top_points[2])
        self.magnet.aperture_width = Beamline.get_distance(top_points[4], top_points[5])

        side_points = Beamline.get_csv('digitize/magnet.side.csv')
        self.magnet.height = Beamline.get_distance(side_points[1], side_points[2])
        self.magnet.aperture_height = Beamline.get_distance(side_points[4], side_points[5])

    def get_collimator_ds_dimension(self):
        top_points = Beamline.get_csv('digitize/collimator_ds.csv')
        self.collimator_ds.length = (Beamline.get_distance(top_points[0], top_points[1]) + Beamline.get_distance(top_points[2], top_points[3])) / 2.
        self.collimator_ds.width = Beamline.get_distance(top_points[1], top_points[2])
        self.collimator_ds.aperture_width = Beamline.get_distance(top_points[4], top_points[5])

        side_points = Beamline.get_csv('digitize/collimator_ds.side.csv')
        self.collimator_ds.height = Beamline.get_distance(side_points[1], side_points[2])
        self.collimator_ds.aperture_height = Beamline.get_distance(side_points[4], side_points[5])

    def get_collimator_us_dimension(self):
        top_points = Beamline.get_csv('digitize/collimator_us.csv')
        self.collimator_ds.length = np.average([Beamline.get_distance(top_points[0], top_points[2]), Beamline.get_distance(top_points[3], top_points[5])])
        self.collimator_ds.width = np.average([Beamline.get_distance(top_points[2], top_points[3]), Beamline.get_distance(top_points[1], top_points[4])])

        side_points = Beamline.get_csv('digitize/collimator_us.side.csv')
        self.collimator_ds.height = np.average([Beamline.get_distance(side_points[0], side_points[5]), Beamline.get_distance(side_points[1], side_points[4])])
        self.collimator_ds.aperture_height = np.average([Beamline.get_distance(side_points[2], side_points[3]), Beamline.get_distance(side_points[6], side_points[7])])

    def plot_position(self):

        colors = [
            kBlack,
            kRed + 2,
            kMagenta + 2,
            kViolet + 2,
            kBlue + 2,
            kAzure + 2,
            kCyan + 2,
            kTeal + 2,
            kGreen + 2,
            kSpring + 2,
            kYellow + 2,
            kOrange + 2
        ]

        styles = [
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31
        ]

        c1 = TCanvas('c1', 'c1', 1200, 600)
        set_margin()
        gPad.SetTickx()
        gPad.SetTicky()

        gr = TGraph(2, np.array([-50., 1500.]), np.array([-1000., 1500.]))
        set_graph_style(gr)
        gr.SetMarkerSize(0)
        gr.SetLineWidth(0)
        gr.GetXaxis().SetTitle('Z (cm)')
        gr.GetYaxis().SetTitle('X (cm)')
        gr.Draw('AP')
        gr.GetXaxis().SetRangeUser(-50, 1600)
        gr.GetYaxis().SetRangeUser(-150, 50)
        gr.GetYaxis().SetNdivisions(505, 1)
        gr.GetXaxis().SetNdivisions(508, 1)

        lg1 = TLegend(0.52, 0.33, 0.87, 0.86)
        set_legend_style(lg1)
        lg1.SetTextSize(22)
        lg1.SetMargin(0.15)
        lg1.SetBorderSize(1)

        markers = []
        for i, detector in enumerate(self.detectors):
            if i == len(self.detectors) - 1:
                continue
            marker = TMarker(detector.z, detector.x, 24)
            markers.append(marker)
            markers[i].SetMarkerColor(colors[i])
            markers[i].SetMarkerStyle(styles[i])
            markers[i].SetMarkerSize(2.)
            markers[i].Draw()

            name = '{} ({:.1f}, {:.1f})'.format(detector.name, detector.z, detector.x)
            lg1.AddEntry(markers[i], name, 'p')

        length = 10.
        nova_detector_line = TLine(self.nova.z, self.nova.x - length, self.nova.z, self.nova.x + length)
        nova_detector_line.SetLineStyle(2)
        nova_detector_line.SetLineWidth(2)
        nova_detector_line.Draw()
        lg1.AddEntry(nova_detector_line, 'NOvA detector front ({:.1f}, {:.1f})'.format(self.nova.z, self.nova.x), 'l')

        lg1.Draw()
        c1.Update()
        c1.SaveAs('{}/plot_position.pdf'.format(self.figure_dir))
        input('Press any key to continue.')


beamline = Beamline()
beamline.figure_dir = 'figures'
beamline.plot_position()
