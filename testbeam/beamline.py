import csv
from rootalias import *
from math import pi, tan, atan, sin, cos
from pprint import pprint


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
        self.theta = 0.
        self.aperture_width = 1.
        self.aperture_height = 1.
        self.color = kBlack
        self.marker_style = 20

    def set_xyz(self, xyz):
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]

    def set_zx(self, zx):
        self.z = zx[0]
        self.x = zx[1]

    def get_r(self):
        return (self.z**2 + self.x**2)**0.5


class Beamline:
    INCH = 25.4                 # mm
    FOOT = 304.8                # mm
    RADIAN_PER_DEGREE = pi / 180.
    COLORS = [kBlack, kBlue, kRed + 1, kMagenta + 2, kGreen + 1, kOrange + 1, kYellow + 2, kPink, kViolet, kAzure + 4, kCyan + 1, kTeal - 7, kBlue - 5]
    MARKER_STYLES = list(range(20, 31)) + [2, 3, 4, 5, 8]

    def __init__(self, g4bl_filename='beamline.py.in'):
        self.g4bl_filename = g4bl_filename
        self.figure_dir = None
        self.us_theta = -16.    # degree
        self.ds_theta = 16.     # degree
        self.f_out = open(g4bl_filename, 'w')
        self.screen_shot = False
        # self.kill = 1
        self.kill = 0
        # self.detector_type = 'virtualdetector'
        self.detector_type = 'detector'
        # self.magnet_by = -0.9    # B field in tesla
        # self.distance_target_to_ground = 83. * Beamline.INCH # estimated height of the beam pipe center with respect to the ground
        # self.wire_chamber_support_length = 12.061 * Beamline.INCH
        # self.grating_upstream_edge_z = 298.211 * Beamline.INCH

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
        self.tof_ds_sipm = Detector('downstream TOF SiPM')
        self.nova = Detector('nova detector')
        self.shielding_block = Detector('shielding block')
        self.shielding_block_1 = Detector('shielding block 1')
        self.shielding_block_2 = Detector('shielding block 2')
        self.shielding_block_3 = Detector('shielding block 3')
        self.helium_pipe_1 = Detector('helium pipe 1')
        self.helium_pipe_2 = Detector('helium pipe 2')
        self.helium_pipe_3 = Detector('helium pipe 3')
        self.helium_pipe_4 = Detector('helium pipe 4')

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
            self.tof_ds_sipm,
            self.nova,
        ]

        self.shielding_blocks = [
            self.shielding_block_1,
            self.shielding_block_2,
            self.shielding_block_3,
        ]

        self.helium_pipes = [
            self.helium_pipe_1,
            self.helium_pipe_2,
            self.helium_pipe_3,
            self.helium_pipe_4,
        ]

        self.components = self.detectors + self.shielding_blocks + self.helium_pipes

        # self.read_position()
        # self.correct_position()
        # self.read_nova_dimension()
        # self.read_magnet_dimension()
        # self.read_collimator_us_dimension()
        # self.read_collimator_ds_dimension()
        # self.read_cherenkov_dimension()

    def __del__(self):
        self.f_out.close()

    # @staticmethod
    # def get_average(points):
    #     dimension_count = len(points[0])
    #     point_count = len(points)
    #     average = []
    #     for i in range(dimension_count):
    #         average.append(sum([point[i] for point in points]) / point_count)
    #     return average

    # @staticmethod
    # def get_distance(point_1, point_2):
    #     dimension_count = len(point_1)
    #     distance = 0.
    #     for i in range(dimension_count):
    #         distance += (point_1[i] - point_2[i])**2
    #     return distance**0.5

    # @staticmethod
    # def get_csv(filename):
    #     rows = []
    #     with open(filename) as f_csv:
    #         csv_reader = csv.reader(f_csv, delimiter=',')
    #         next(csv_reader, None)
    #         for row in csv_reader:
    #             rows.append(list(map(lambda x: float(x) * 10., row)))
    #     return rows

    # def read_position(self):
    #     rows = Beamline.get_csv('digitize/ftbf_drawing_digitize.csv')
    #     origin = rows[0]
    #     rows = [[row[0] - origin[0], row[1] - origin[1]] for row in rows]

    #     self.target.set_zx(rows[0])
    #     self.tof_us.set_zx(rows[1])
    #     self.wc_1.set_zx(Beamline.get_average([rows[2], rows[3]]))
    #     self.wc_2.set_zx(Beamline.get_average([rows[4], rows[5]]))
    #     self.magnet.set_zx(Beamline.get_average([rows[6], rows[9]]))
    #     self.wc_3.set_zx(Beamline.get_average([rows[10], rows[11]]))
    #     self.collimator_ds.set_zx(Beamline.get_average([rows[12], rows[13]]))
    #     self.wc_4.set_zx(Beamline.get_average([rows[14], rows[15]]))
    #     self.cherenkov.set_zx(Beamline.get_average([rows[16], rows[17]]))
    #     self.tof_ds.set_zx(rows[18])
    #     self.nova.set_zx(rows[19])

    #     collimator_us_points = Beamline.get_csv('digitize/collimator_us.csv')
    #     collimator_us_position = Beamline.get_average([collimator_us_points[0], collimator_us_points[2], collimator_us_points[3], collimator_us_points[5]])
    #     self.collimator_us.set_zx([collimator_us_position[0] - origin[0], collimator_us_position[1] - origin[1]])

    # def read_nova_dimension(self):
    #     top_points = Beamline.get_csv('digitize/nova.csv')
    #     self.nova.length = np.average([Beamline.get_distance(top_points[0], top_points[1]), Beamline.get_distance(top_points[2], top_points[3])])
    #     self.nova.width = Beamline.get_distance(top_points[1], top_points[2])
    #     self.nova.height = self.nova.width

    # def read_magnet_dimension(self):
    #     top_points = Beamline.get_csv('digitize/magnet.csv')
    #     self.magnet.length = np.average([Beamline.get_distance(top_points[0], top_points[1]), Beamline.get_distance(top_points[2], top_points[3])])
    #     self.magnet.width = Beamline.get_distance(top_points[1], top_points[2])
    #     self.magnet.aperture_width = Beamline.get_distance(top_points[4], top_points[5])

    #     side_points = Beamline.get_csv('digitize/magnet.side.csv')
    #     self.magnet.height = Beamline.get_distance(side_points[1], side_points[2])
    #     self.magnet.aperture_height = Beamline.get_distance(side_points[4], side_points[5])

    # def read_collimator_ds_dimension(self):
    #     top_points = Beamline.get_csv('digitize/collimator_ds.csv')
    #     self.collimator_ds.length = (Beamline.get_distance(top_points[0], top_points[1]) + Beamline.get_distance(top_points[2], top_points[3])) / 2.
    #     self.collimator_ds.width = Beamline.get_distance(top_points[1], top_points[2])
    #     self.collimator_ds.aperture_width = Beamline.get_distance(top_points[4], top_points[5])

    #     side_points = Beamline.get_csv('digitize/collimator_ds.side.csv')
    #     self.collimator_ds.height = Beamline.get_distance(side_points[1], side_points[2])
    #     self.collimator_ds.aperture_height = Beamline.get_distance(side_points[4], side_points[5])

    # def read_collimator_us_dimension(self):
    #     top_points = Beamline.get_csv('digitize/collimator_us.csv')
    #     self.collimator_ds.length = np.average([Beamline.get_distance(top_points[0], top_points[2]), Beamline.get_distance(top_points[3], top_points[5])])
    #     self.collimator_ds.width = np.average([Beamline.get_distance(top_points[2], top_points[3]), Beamline.get_distance(top_points[1], top_points[4])])

    #     side_points = Beamline.get_csv('digitize/collimator_us.side.csv')
    #     self.collimator_ds.height = np.average([Beamline.get_distance(side_points[0], side_points[5]), Beamline.get_distance(side_points[1], side_points[4])])
    #     self.collimator_ds.aperture_height = np.average([Beamline.get_distance(side_points[2], side_points[3]), Beamline.get_distance(side_points[6], side_points[7])])

    # def read_cherenkov_dimension(self):
    #     top_points = Beamline.get_csv('digitize/ftbf_drawing_digitize.csv')
    #     self.cherenkov.length = self.get_distance(top_points[16], top_points[17])

    # def correct_position(self):
    #     us_detectors = [
    #         self.tof_us,
    #         self.wc_1,
    #         self.wc_2
    #     ]
    #     for us_detector in us_detectors:
    #         distance = us_detector.get_r()
    #         us_detector.x = sin(self.us_theta * Beamline.RADIAN_PER_DEGREE) * distance
    #         us_detector.z = cos(self.us_theta * Beamline.RADIAN_PER_DEGREE) * distance

    #     ds_detectors = [
    #         self.wc_3,
    #         self.collimator_ds,
    #         self.wc_4,
    #         self.cherenkov,
    #         self.tof_ds,
    #         self.nova
    #     ]
    #     ds_detector_average_x = np.average([ds_detector.x for ds_detector in ds_detectors])
    #     for ds_detector in ds_detectors:
    #         ds_detector.x = ds_detector_average_x

    #     self.magnet.x = ds_detector_average_x
    #     self.magnet.z = ds_detector_average_x / tan(self.us_theta * Beamline.RADIAN_PER_DEGREE)

    def plot_position(self):
        c1 = TCanvas('c1', 'c1', 1400, 800)
        set_margin()
        gPad.SetLeftMargin(0.1)
        gPad.SetRightMargin(0.1)
        gPad.SetTickx()
        gPad.SetTicky()

        gr = TGraph(2, np.array([-50., 1500.]), np.array([-1000., 1500.]))
        set_graph_style(gr)
        gr.SetMarkerSize(0)
        gr.SetLineWidth(0)
        gr.GetXaxis().SetTitle('Z (cm)')
        gr.GetYaxis().SetTitle('X (cm)')
        gr.GetYaxis().SetTitleOffset(1.)
        gr.Draw('AP')
        gr.GetXaxis().SetRangeUser(-50, 1600)
        gr.GetYaxis().SetRangeUser(-220, 120)
        gr.GetYaxis().SetNdivisions(505, 1)
        gr.GetXaxis().SetNdivisions(508, 1)

        lg1 = TLegend(0.36, 0.53, 0.87, 0.86)
        set_legend_style(lg1)
        lg1.SetNColumns(2)
        lg1.SetTextSize(18)
        lg1.SetMargin(0.15)
        lg1.SetBorderSize(1)

        markers = []
        for i, detector in enumerate(self.components):
            marker = TMarker(detector.z / 10., detector.x / 10., 24)
            markers.append(marker)
            markers[i].SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])
            markers[i].SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            markers[i].SetMarkerSize(2.)
            if detector.name != 'nova detector':
                markers[i].Draw()
                name = '{} ({:.1f}, {:.1f})'.format(detector.name, detector.z / 10., detector.x / 10.)
                lg1.AddEntry(markers[i], name, 'p')

        length = 20.
        nova_detector_line = TLine(self.nova.z / 10., self.nova.x / 10. - length, self.nova.z / 10., self.nova.x / 10. + length)
        nova_detector_line.SetLineStyle(2)
        nova_detector_line.SetLineWidth(2)
        nova_detector_line.Draw()
        lg1.AddEntry(nova_detector_line, 'NOvA detector front ({:.1f}, {:.1f})'.format(self.nova.z / 10., self.nova.x / 10.), 'l')

        lg1.Draw()
        c1.Update()
        c1.SaveAs('{}/plot_position.pdf'.format(self.figure_dir))
        input('Press any key to continue.')

    def print_flight_distance(self):
        detectors = [
            self.tof_us,
            self.wc_1,
            self.wc_2,
            self.magnet,
            self.wc_3,
            self.collimator_ds,
            self.wc_4,
            self.cherenkov,
            self.tof_ds,
            self.tof_ds_sipm,
        ]

        flight_distance = 0.
        last_detector = detectors[0]
        for i, detector in enumerate(detectors[1:]):
            length = ((detector.x - last_detector.x)**2 + (detector.z - last_detector.z)**2)**0.5
            flight_distance += length
            last_detector = detector
            print('detector.name = {:25s} flight_distance = {}'.format(detector.name, flight_distance))
        print('from tof_us to tof_ds_sipm: {} mm'.format(flight_distance))

        flight_distance = 0.
        last_detector = detectors[0]
        for i, detector in enumerate(detectors[1:-1]):
            length = ((detector.x - last_detector.x)**2 + (detector.z - last_detector.z)**2)**0.5
            flight_distance += length
            last_detector = detector
            print('detector.name = {:25s} flight_distance = {}'.format(detector.name, flight_distance))
        print('from tof_us to tof_ds: {} mm'.format(flight_distance))

    def write_target(self):
        target_slab_dimensions = [31.75, 209.55, 6.35]  # [height, length, width]
        target_slab_count = 5.
        target_delta_x = target_slab_dimensions[0] / target_slab_count
        target_delta_z = 22.145
        self.f_out.write('box slab height={} length={} width={} material=Cu color=1,0.01,0.01\n'.format(target_slab_dimensions[0], target_slab_dimensions[1], target_slab_dimensions[2]))
        for i in range(-2, 3):
            self.f_out.write('place slab rename=target_slab_{} x={} y={} z={}\n'.format(i, self.target.x + i * target_delta_x, self.target.y, self.target.z - i * target_delta_z))

    def write_collimator_us(self):
        collimator_us_base_dimensions = [5.19 * Beamline.INCH, 58. * Beamline.INCH, 32. * Beamline.INCH]
        collimator_us_plate_dimensions = [5.19 * Beamline.INCH, 42.76 * Beamline.INCH, 32. * Beamline.INCH]
        collimator_us_aperture_dimensions = [2. * Beamline.INCH, 42.76 * Beamline.INCH, 11.6 * Beamline.INCH]

        collimator_us_base_positions = [0., -(1. + 5.19 * 1.5) * Beamline.INCH, 0.]
        collimator_us_bottom_positions = [0., -(1. + 5.19 / 2.) * Beamline.INCH, 7.62 * Beamline.INCH]
        collimator_us_aperture_1_positions = [296. / 2. + 67.29, 0., 7.62 * Beamline.INCH]
        collimator_us_aperture_2_positions = [-296. / 2. - 67.29, 0., 7.62 * Beamline.INCH]
        collimator_us_top_1_positions = [0., (1. + 5.19 / 2.) * Beamline.INCH, 7.62 * Beamline.INCH]
        collimator_us_top_2_positions = [0., (1. + 5.19 * 1.5) * Beamline.INCH, 7.62 * Beamline.INCH]

        collimator_us_z_offset = (29. / 2. + 7.62) * Beamline.INCH
        collimator_us_theta_offset = 1.97
        collimator_us_aperture_1_theta = self.us_theta + collimator_us_theta_offset
        collimator_us_aperture_2_theta = self.us_theta - collimator_us_theta_offset

        collimator_us_parts = [
            collimator_us_base_positions,
            collimator_us_bottom_positions,
            collimator_us_aperture_1_positions,
            collimator_us_aperture_2_positions,
            collimator_us_top_1_positions,
            collimator_us_top_2_positions
        ]
        for collimator_us_part in collimator_us_parts:
            collimator_us_part[2] += collimator_us_z_offset
        x_offset = tan(self.us_theta * pi / 180.) * collimator_us_aperture_1_positions[2]
        for collimator_us_part in collimator_us_parts:
            collimator_us_part[0] += x_offset

        self.collimator_us.length = collimator_us_base_dimensions[1]
        self.collimator_us.width = collimator_us_base_dimensions[2]

        self.f_out.write('box collimator_us_base height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_us_base_dimensions[0], collimator_us_base_dimensions[1], collimator_us_base_dimensions[2], self.kill))
        self.f_out.write('box collimator_us_plate height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_us_plate_dimensions[0], collimator_us_plate_dimensions[1], collimator_us_plate_dimensions[2], self.kill))
        self.f_out.write('box collimator_us_aperture height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_us_aperture_dimensions[0], collimator_us_aperture_dimensions[1], collimator_us_aperture_dimensions[2], self.kill))

        self.f_out.write('place collimator_us_base rename=collimator_us_base x={} y={} z={}\n'.format(collimator_us_base_positions[0], collimator_us_base_positions[1], collimator_us_base_positions[2]))
        self.f_out.write('place collimator_us_plate rename=collimator_us_bottom x={} y={} z={}\n'.format(collimator_us_bottom_positions[0], collimator_us_bottom_positions[1], collimator_us_bottom_positions[2]))
        self.f_out.write('place collimator_us_aperture rename=collimator_us_aperture_1 x={} y={} z={} rotation=y{}\n'.format(collimator_us_aperture_1_positions[0], collimator_us_aperture_1_positions[1], collimator_us_aperture_1_positions[2], collimator_us_aperture_1_theta))
        self.f_out.write('place collimator_us_aperture rename=collimator_us_aperture_2 x={} y={} z={} rotation=y{}\n'.format(collimator_us_aperture_2_positions[0], collimator_us_aperture_2_positions[1], collimator_us_aperture_2_positions[2], collimator_us_aperture_2_theta))
        self.f_out.write('place collimator_us_plate rename=collimator_us_top_1 x={} y={} z={}\n'.format(collimator_us_top_1_positions[0], collimator_us_top_1_positions[1], collimator_us_top_1_positions[2]))
        self.f_out.write('place collimator_us_plate rename=collimator_us_top_2 x={} y={} z={}\n'.format(collimator_us_top_2_positions[0], collimator_us_top_2_positions[1], collimator_us_top_2_positions[2]))

    def write_collimator_us_alignment(self):
        # all numbers are in inch
        # alignment
        # NTB-TGT-COLL-002-CHANNEL_UP
        channel_up_x = -2.215
        channel_up_y = -0.040
        channel_up_z = 7.694
        # NTB-TGT-COLL-002-CHANNEL_DN
        channel_down_x = -14.561
        channel_down_y = 0.012
        channel_down_z = 50.439
        # drawing by J. Kilmer
        aperture_plate_up_edge_small = 9.02
        aperture_plate_up_edge_large = 20.07
        aperture_plate_height = 2.
        collimator_width = 32.
        collimator_base_height = 5.19
        collimator_base_length = 58.
        collimator_plate_height = 5.19
        # measurement by M. Vincent
        channel_down_width = 6.

        channel_up_width = collimator_width - aperture_plate_up_edge_small - aperture_plate_up_edge_large
        aperture_plate_length = channel_down_z - channel_up_z
        aperture_plate_y = (channel_up_y + channel_down_y) / 2.

        aperture_plate_1_vertex_1_x = channel_up_x + channel_up_width / 2. + aperture_plate_up_edge_small
        aperture_plate_1_vertex_1_z = channel_up_z
        aperture_plate_1_vertex_2_x = channel_up_x + channel_up_width / 2.
        aperture_plate_1_vertex_2_z = aperture_plate_1_vertex_1_z
        aperture_plate_1_vertex_3_x = channel_down_x + channel_down_width / 2.
        aperture_plate_1_vertex_3_z = channel_down_z
        aperture_plate_1_vertex_4_x = aperture_plate_1_vertex_1_x
        aperture_plate_1_vertex_4_z = aperture_plate_1_vertex_3_z
        self.f_out.write('extrusion aperture_plate_1 length={} vertices={},{};{},{};{},{};{},{} material=Fe color=0,1,1 kill={}\n'.format(aperture_plate_height * Beamline.INCH,
                                                                                                                                          0, 0,
                                                                                                                                          (aperture_plate_1_vertex_2_x - aperture_plate_1_vertex_1_x) * Beamline.INCH, (aperture_plate_1_vertex_2_z - aperture_plate_1_vertex_1_z) * Beamline.INCH,
                                                                                                                                          (aperture_plate_1_vertex_3_x - aperture_plate_1_vertex_1_x) * Beamline.INCH, (aperture_plate_1_vertex_3_z - aperture_plate_1_vertex_1_z) * Beamline.INCH,
                                                                                                                                          (aperture_plate_1_vertex_4_x - aperture_plate_1_vertex_1_x) * Beamline.INCH, (aperture_plate_1_vertex_4_z - aperture_plate_1_vertex_1_z) * Beamline.INCH,
                                                                                                                                          self.kill))
        self.f_out.write('place aperture_plate_1 rename=aperture_plate_1 x={} y={} z={} rotation=x90\n'.format(aperture_plate_1_vertex_1_x * Beamline.INCH, aperture_plate_y * Beamline.INCH, aperture_plate_1_vertex_1_z * Beamline.INCH))

        aperture_plate_2_vertex_1_x = channel_up_x - channel_up_width / 2.
        aperture_plate_2_vertex_1_z = channel_up_z
        aperture_plate_2_vertex_2_x = channel_up_x - channel_up_width / 2. - aperture_plate_up_edge_large
        aperture_plate_2_vertex_2_z = aperture_plate_2_vertex_1_z
        aperture_plate_2_vertex_3_x = aperture_plate_2_vertex_2_x
        aperture_plate_2_vertex_3_z = channel_down_z
        aperture_plate_2_vertex_4_x = channel_down_x - channel_down_width / 2.
        aperture_plate_2_vertex_4_z = aperture_plate_2_vertex_3_z
        self.f_out.write('extrusion aperture_plate_2 length={} vertices={},{};{},{};{},{};{},{} material=Fe color=0,1,1 kill={}\n'.format(aperture_plate_height * Beamline.INCH,
                                                                                                                                          0, 0,
                                                                                                                                          (aperture_plate_2_vertex_2_x - aperture_plate_2_vertex_1_x) * Beamline.INCH, (aperture_plate_2_vertex_2_z - aperture_plate_2_vertex_1_z) * Beamline.INCH,
                                                                                                                                          (aperture_plate_2_vertex_3_x - aperture_plate_2_vertex_1_x) * Beamline.INCH, (aperture_plate_2_vertex_3_z - aperture_plate_2_vertex_1_z) * Beamline.INCH,
                                                                                                                                          (aperture_plate_2_vertex_4_x - aperture_plate_2_vertex_1_x) * Beamline.INCH, (aperture_plate_2_vertex_4_z - aperture_plate_2_vertex_1_z) * Beamline.INCH,
                                                                                                                                          self.kill))
        self.f_out.write('place aperture_plate_2 rename=aperture_plate_2 x={} y={} z={} rotation=x90\n'.format(aperture_plate_2_vertex_1_x * Beamline.INCH, aperture_plate_y * Beamline.INCH, aperture_plate_2_vertex_1_z * Beamline.INCH))

        aperture_plate_x = (aperture_plate_1_vertex_1_x + aperture_plate_2_vertex_2_x) / 2.
        aperture_plate_z = (aperture_plate_1_vertex_1_z + aperture_plate_1_vertex_4_z) / 2.

        collimator_plate_length = aperture_plate_length
        collimator_base_dimensions = [collimator_base_height, collimator_base_length, collimator_width]
        collimator_plate_dimensions = [collimator_plate_height, collimator_plate_length, collimator_width]

        collimator_base_positions = [aperture_plate_x, aperture_plate_y - 0.5 * aperture_plate_height - collimator_plate_dimensions[0] - 0.5 * collimator_base_dimensions[0], aperture_plate_z - (collimator_base_dimensions[1] - collimator_plate_dimensions[1]) / 2.]
        collimator_bottom_positions = [aperture_plate_x, aperture_plate_y - 0.5 * aperture_plate_height - 0.5 * collimator_plate_dimensions[0], aperture_plate_z]
        collimator_top_1_positions = [aperture_plate_x, aperture_plate_y + 0.5 * aperture_plate_height + 0.5 * collimator_plate_dimensions[0], aperture_plate_z]
        collimator_top_2_positions = [aperture_plate_x, aperture_plate_y + 0.5 * aperture_plate_height + 1.5 * collimator_plate_dimensions[0], aperture_plate_z]

        for quantity in [collimator_base_dimensions, collimator_plate_dimensions, collimator_base_positions, collimator_bottom_positions, collimator_top_1_positions, collimator_top_2_positions]:
            quantity[0] *= Beamline.INCH
            quantity[1] *= Beamline.INCH
            quantity[2] *= Beamline.INCH

        self.collimator_us.set_xyz([aperture_plate_x * Beamline.INCH, aperture_plate_y * Beamline.INCH, aperture_plate_z * Beamline.INCH])

        self.f_out.write('box collimator_base height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_base_dimensions[0], collimator_base_dimensions[1], collimator_base_dimensions[2], self.kill))
        self.f_out.write('box collimator_plate height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_plate_dimensions[0], collimator_plate_dimensions[1], collimator_plate_dimensions[2], self.kill))
        self.f_out.write('place collimator_base rename=collimator_base x={} y={} z={}\n'.format(collimator_base_positions[0], collimator_base_positions[1], collimator_base_positions[2]))
        self.f_out.write('place collimator_plate rename=collimator_bottom x={} y={} z={}\n'.format(collimator_bottom_positions[0], collimator_bottom_positions[1], collimator_bottom_positions[2]))
        self.f_out.write('place collimator_plate rename=collimator_top_1 x={} y={} z={}\n'.format(collimator_top_1_positions[0], collimator_top_1_positions[1], collimator_top_1_positions[2]))
        self.f_out.write('place collimator_plate rename=collimator_top_2 x={} y={} z={}\n'.format(collimator_top_2_positions[0], collimator_top_2_positions[1], collimator_top_2_positions[2]))

    def write_virtual_disk(self):
        start_line_radius = 1750.
        start_line_length = 1.
        z_shift = 50.           # distance upstream of the upstream tof
        self.f_out.write('{} start_line radius={} length={} material=Air color=0.9,0.9,0.7\n'.format(self.detector_type, start_line_radius, start_line_length))
        self.f_out.write('place start_line x={} y={} z={}\n'.format(self.tof_us.x, self.tof_us.y, self.tof_us.z - z_shift))

    def write_tof(self):
        tof_us_dimensions = [5.91 * Beamline.INCH, 0.79 * Beamline.INCH, 5.91 * Beamline.INCH]
        self.tof_us.theta = self.us_theta
        tof_ds_dimensions = [5.91 * Beamline.INCH, 0.24 * Beamline.INCH, 5.91 * Beamline.INCH]
        self.tof_ds.theta = self.us_theta + self.ds_theta
        tof_ds_sipm_dimensions = [5.91 * Beamline.INCH, 0.52 * Beamline.INCH, 5.91 * Beamline.INCH]
        self.tof_ds_sipm.theta = self.us_theta + self.ds_theta

        self.f_out.write('{} tof_us  height={} length={} width={} material=LUCITE color=0,0,1\n'.format(self.detector_type, tof_us_dimensions[0], tof_us_dimensions[1], tof_us_dimensions[2]))
        self.f_out.write('place tof_us rename=tof_us x={} y={} z={} rotation=y{}\n'.format(self.tof_us.x, self.tof_us.y, self.tof_us.z, self.tof_us.theta))
        self.f_out.write('{} tof_ds height={} length={} width={} material=LUCITE color=0,0,1\n'.format(self.detector_type, tof_ds_dimensions[0], tof_ds_dimensions[1], tof_ds_dimensions[2]))
        self.f_out.write('place tof_ds rename=tof_ds x={} y={} z={} rotation=y{}\n'.format(self.tof_ds.x, self.tof_ds.y, self.tof_ds.z, self.tof_ds.theta))
        self.f_out.write('{} tof_ds_sipm height={} length={} width={} material=LUCITE color=0,0,1\n'.format(self.detector_type, tof_ds_sipm_dimensions[0], tof_ds_sipm_dimensions[1], tof_ds_sipm_dimensions[2]))
        self.f_out.write('place tof_ds_sipm rename=tof_ds_sipm x={} y={} z={} rotation=y{}\n'.format(self.tof_ds_sipm.x, self.tof_ds_sipm.y, self.tof_ds_sipm.z, self.tof_ds_sipm.theta))

    def write_wc(self):
        wire_chamber_detector_dimensions = [128., 25., 128.]
        wire_chamber_frame_vertical_dimensions = [254., 25., 63.]
        wire_chamber_frame_horizontal_dimensions = [63., 25., 128.]
        wire_chamber_detector_positions = [0., 0., 12.5]
        wire_chamber_frame_vertical_left_positions = [-95.5, 0., 12.5]
        wire_chamber_frame_vertical_right_positions = [95.5, 0., 12.5]
        wire_chamber_frame_horizontal_top_positions = [0., 95.5, 12.5]
        wire_chamber_frame_horizontal_bottom_positions = [0., -95.5, 12.5]
        self.wc_1.theta = self.us_theta
        self.wc_2.theta = self.us_theta
        self.wc_3.theta = self.us_theta + self.ds_theta
        self.wc_4.theta = self.us_theta + self.ds_theta

        # place wire chamber 3 by the upstream edge of the grating
        # self.wc_3.z = self.grating_upstream_edge_z - self.wire_chamber_support_length / 2.

        self.f_out.write('group wire_chamber\n')
        self.f_out.write('  {} wire_chamber_detector height={} length={} width={} material=Ar color=0,1,0\n'.format(self.detector_type, wire_chamber_detector_dimensions[0], wire_chamber_detector_dimensions[1], wire_chamber_detector_dimensions[2]))
        self.f_out.write('  box wire_chamber_frame_vertical height={} length={} width={} color=1,0,1 kill={} material=Al\n'.format(wire_chamber_frame_vertical_dimensions[0], wire_chamber_frame_vertical_dimensions[1], wire_chamber_frame_vertical_dimensions[2], self.kill))
        self.f_out.write('  box wire_chamber_frame_horizontal height={} length={} width={} color=1,0,1 kill={} material=Al\n'.format(wire_chamber_frame_horizontal_dimensions[0], wire_chamber_frame_horizontal_dimensions[1], wire_chamber_frame_horizontal_dimensions[2], self.kill))
        self.f_out.write('  place wire_chamber_frame_vertical rename=+_frame_left x={} y={} z={}\n'.format(wire_chamber_frame_vertical_left_positions[0], wire_chamber_frame_vertical_left_positions[1], wire_chamber_frame_vertical_left_positions[2]))
        self.f_out.write('  place wire_chamber_frame_vertical rename=+_frame_right x={} y={} z={}\n'.format(wire_chamber_frame_vertical_right_positions[0], wire_chamber_frame_vertical_right_positions[1], wire_chamber_frame_vertical_right_positions[2]))
        self.f_out.write('  place wire_chamber_frame_horizontal rename=+_frame_top x={} y={} z={}\n'.format(wire_chamber_frame_horizontal_top_positions[0], wire_chamber_frame_horizontal_top_positions[1], wire_chamber_frame_horizontal_top_positions[2]))
        self.f_out.write('  place wire_chamber_frame_horizontal rename=+_frame_bottom x={} y={} z={}\n'.format(wire_chamber_frame_horizontal_bottom_positions[0], wire_chamber_frame_horizontal_bottom_positions[1], wire_chamber_frame_horizontal_bottom_positions[2]))
        self.f_out.write('  place wire_chamber_detector rename=+_detector x={} y={} z={}\n'.format(wire_chamber_detector_positions[0], wire_chamber_detector_positions[1], wire_chamber_detector_positions[2]))
        self.f_out.write('endgroup\n')
        self.f_out.write('place wire_chamber rename=wire_chamber_1 x={} y={} z={} rotation=y{}\n'.format(self.wc_1.x, self.wc_1.y, self.wc_1.z, self.wc_1.theta))
        self.f_out.write('place wire_chamber rename=wire_chamber_2 x={} y={} z={} rotation=y{}\n'.format(self.wc_2.x, self.wc_2.y, self.wc_2.z, self.wc_2.theta))
        self.f_out.write('place wire_chamber rename=wire_chamber_3 x={} y={} z={} rotation=y{}\n'.format(self.wc_3.x, self.wc_3.y, self.wc_3.z, self.wc_3.theta))
        self.f_out.write('place wire_chamber rename=wire_chamber_4 x={} y={} z={} rotation=y{}\n'.format(self.wc_4.x, self.wc_4.y, self.wc_4.z, self.wc_4.theta))

    def write_helium_pipe(self):
        self.helium_pipe_1.theta = self.us_theta
        self.helium_pipe_2.theta = self.us_theta + self.ds_theta
        self.helium_pipe_3.theta = self.us_theta + self.ds_theta
        self.helium_pipe_4.theta = self.us_theta + self.ds_theta

        outer_radius = 6. * Beamline.INCH / 2.
        wall_thickness = 3. / 32. * Beamline.INCH
        inner_radius = outer_radius - wall_thickness
        mylar_window_thickness = 0.003 * Beamline.INCH

        self.f_out.write('tubs mylar_window radius={} length={} color=0,0.8,0 material=MYLAR\n'.format(outer_radius, mylar_window_thickness))
        for i, helium_pipe in enumerate(self.helium_pipes):
            index = i + 1
            self.f_out.write('group helium_pipe_{}\n'.format(index))
            self.f_out.write('  tubs helium radius={} length={} color=1,1,1 material=He\n'.format(inner_radius, helium_pipe.length))
            self.f_out.write('  tubs helium_pipe innerRadius={} outerRadius={} length={} color=0,0.8,0 material=STAINLESS-STEEL kill={}\n'.format(inner_radius, outer_radius, helium_pipe.length, self.kill))

            z_shift = (helium_pipe.length + mylar_window_thickness * 2.) / 2. # shift in z by half of the full length in z to avoid geometry conflict in the group
            self.f_out.write('  place helium_pipe rename=helium_pipe x={} y={} z={}\n'.format(0., 0., z_shift))
            self.f_out.write('  place helium rename=helium x={} y={} z={}\n'.format(0., 0., z_shift))
            self.f_out.write('  place mylar_window rename=mylar_window_up x={} y={} z={}\n'.format(0., 0., -helium_pipe.length / 2. - mylar_window_thickness / 2. + z_shift))
            self.f_out.write('  place mylar_window rename=mylar_window_down x={} y={} z={}\n'.format(0., 0., helium_pipe.length / 2. + mylar_window_thickness / 2. + z_shift))
            self.f_out.write('endgroup\n')
            self.f_out.write('place helium_pipe_{} rename=helium_pipe_{} x={} y={} z={} rotation=y{}\n'.format(index, index, helium_pipe.x, helium_pipe.y, helium_pipe.z, helium_pipe.theta))

    def write_magnet(self):
        self.magnet.height = 30. * Beamline.INCH

        magnet_field_dimensions = [3.5 * Beamline.INCH, 42 * Beamline.INCH, 17.75 * Beamline.INCH]
        magnet_iron_dimensions = [self.magnet.height, 42 * Beamline.INCH, 50 * Beamline.INCH]
        self.magnet.theta = self.us_theta + self.ds_theta / 2.

        self.f_out.write('genericbend M1 fieldHeight={} fieldLength={} fieldWidth={} kill={} ironColor=1,0,0 ironHeight={} ironLength={} ironWidth={}\n'.format(magnet_field_dimensions[0], magnet_field_dimensions[1], magnet_field_dimensions[2], self.kill, magnet_iron_dimensions[0], magnet_iron_dimensions[1], magnet_iron_dimensions[2]))
        self.f_out.write('place M1 By=$b_field x={} y={} z={} rotation=Y{}\n'.format(self.magnet.x, self.magnet.y, self.magnet.z, self.magnet.theta))

    def write_collimator_ds(self):
        # lariat
        # collimator_ds_bottom_dimensions = [8.5 * Beamline.INCH, 36. * Beamline.INCH, 30. * Beamline.INCH]
        # collimator_ds_middle_dimensions = [6. * Beamline.INCH, 36. * Beamline.INCH, 11. * Beamline.INCH]
        # collimator_ds_middle_1_positions = [9. * Beamline.INCH, 0., 18. * Beamline.INCH]
        # collimator_ds_middle_2_positions = [-9. * Beamline.INCH, 0., 18. * Beamline.INCH]
        # collimator_ds_bottom_positions = [0., (4.25 + 3.) * Beamline.INCH, 18. * Beamline.INCH]
        # collimator_ds_top_positions = [0., -(4.25 + 3.) * Beamline.INCH, 18. * Beamline.INCH]

        aperture_width = 6.25 * Beamline.INCH
        aperture_height = 8.25 * Beamline.INCH
        collimator_width = 28. * Beamline.INCH
        collimator_length = 36. * Beamline.INCH
        collimator_height = 24. * Beamline.INCH
        collimator_ds_bottom_dimensions = [(collimator_height - aperture_height) / 2., collimator_length, collimator_width]
        collimator_ds_middle_dimensions = [aperture_height, collimator_length, (collimator_width - aperture_width) / 2.]
        collimator_ds_middle_1_positions = [aperture_width / 2. + collimator_ds_middle_dimensions[2] / 2., 0., collimator_length / 2.]
        collimator_ds_middle_2_positions = [-aperture_width / 2. - collimator_ds_middle_dimensions[2] / 2., 0., collimator_length / 2.]
        collimator_ds_bottom_positions = [0., aperture_height / 2 + collimator_ds_bottom_dimensions[0] / 2., collimator_length / 2.]
        collimator_ds_top_positions = [0., -aperture_height / 2 - collimator_ds_bottom_dimensions[0] / 2., collimator_length / 2.]

        self.collimator_ds.theta = self.us_theta + self.ds_theta
        self.collimator_ds.width = collimator_width
        self.collimator_ds.length = collimator_length
        self.collimator_ds.height = collimator_height

        self.f_out.write('group collimator_ds\n')
        self.f_out.write('  box collimator_ds_bottom height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_ds_bottom_dimensions[0], collimator_ds_bottom_dimensions[1], collimator_ds_bottom_dimensions[2], self.kill))
        self.f_out.write('  box collimator_ds_middle height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_ds_middle_dimensions[0], collimator_ds_middle_dimensions[1], collimator_ds_middle_dimensions[2], self.kill))
        self.f_out.write('  place collimator_ds_middle rename=+_middle_1 x={} y={} z={}\n'.format(collimator_ds_middle_1_positions[0], collimator_ds_middle_1_positions[1], collimator_ds_middle_1_positions[2]))
        self.f_out.write('  place collimator_ds_middle rename=+_middle_2 x={} y={} z={}\n'.format(collimator_ds_middle_2_positions[0], collimator_ds_middle_2_positions[1], collimator_ds_middle_2_positions[2]))
        self.f_out.write('  place collimator_ds_bottom rename=+_bottom x={} y={} z={}\n'.format(collimator_ds_bottom_positions[0], collimator_ds_bottom_positions[1], collimator_ds_bottom_positions[2]))
        self.f_out.write('  place collimator_ds_bottom rename=+_top x={} y={} z={}\n'.format(collimator_ds_top_positions[0], collimator_ds_top_positions[1], collimator_ds_top_positions[2]))
        self.f_out.write('endgroup\n')
        self.f_out.write('place collimator_ds x={} y={} z={} rotation=y{}\n'.format(self.collimator_ds.x, self.collimator_ds.y, self.collimator_ds.z, self.collimator_ds.theta))

    def write_cherenkov(self):
        # alignment
        self.cherenkov.x = -53.292 # in
        self.cherenkov.y = 0.044   # in
        self.cherenkov.z = 546.554 # in, flange position
        self.cherenkov.z += 1. + 14.75 - 115.18 / 2. # in, convert flange position to geometric center
        self.cherenkov.x *= Beamline.INCH
        self.cherenkov.y *= Beamline.INCH
        self.cherenkov.z *= Beamline.INCH

        self.cherenkov.theta = self.us_theta + self.ds_theta
        self.cherenkov.length = 115.18 * Beamline.INCH
        cherenkov_inner_radius = 315. / 2.
        cherenkov_outer_radius = 324. / 2.
        cherenkov_pmt_pipe_outer_radius = 273. / 2.
        cherenkov_pmt_pipe_inner_radius = 264. / 2.
        cherenkov_pmt_pipe_length = 846. - 324. / 2.
        support_dimensions = [4.5 * Beamline.FOOT, 6. * Beamline.FOOT, 3. * Beamline.FOOT]

        # the distance between wire chamber 4 and the upstream end of the Cerenkov counter is about 48 inches
        # self.cherenkov.z = self.wc_4.z + 48. * Beamline.INCH + self.cherenkov.length / 2.
        cherenkov_end_to_support_edge_distance = 3.625 * Beamline.INCH
        cherenkov_to_support_distance = 6.125 * Beamline.INCH

        self.f_out.write('{} cherenkov radius={} length={} color=1,1,1 material=CARBON_DIOXIDE\n'.format(self.detector_type, cherenkov_inner_radius, self.cherenkov.length))
        self.f_out.write('tubs cherenkov_pipe innerRadius={} outerRadius={} length={} color=0.74,0.34,0.09 material=STAINLESS-STEEL\n'.format(cherenkov_inner_radius, cherenkov_outer_radius, self.cherenkov.length))
        self.f_out.write('tubs cherenkov_pipe_pmt innerRadius={} outerRadius={} length={} color=0.74,0.34,0.09 material=STAINLESS-STEEL\n'.format(cherenkov_pmt_pipe_inner_radius, cherenkov_pmt_pipe_outer_radius, cherenkov_pmt_pipe_length))
        self.f_out.write('box cherenkov_support height={} length={} width={} material=CONCRETE color=0,1,1 kill={}\n'.format(support_dimensions[0], support_dimensions[1], support_dimensions[2], self.kill))

        self.f_out.write('place cherenkov rename=cherenkov x={} y={} z={} rotation=y{}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z, self.cherenkov.theta))
        self.f_out.write('place cherenkov_pipe rename=cherenkov_pipe x={} y={} z={} rotation=y{} kill={}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z, self.cherenkov.theta, self.kill))
        self.f_out.write('place cherenkov_pipe_pmt rename=cherenkov_pipe_pmt x={} y={} z={} rotation=y{},x90 kill={}\n'.format(self.cherenkov.x, self.cherenkov.y - cherenkov_outer_radius - cherenkov_pmt_pipe_length / 2., self.cherenkov.z + self.cherenkov.length / 2. - cherenkov_pmt_pipe_outer_radius - 1.75 * Beamline.INCH, self.cherenkov.theta, self.kill))
        self.f_out.write('place cherenkov_support x={} y={} z={} rotation=y{}\n'.format(self.cherenkov.x, self.cherenkov.y - cherenkov_outer_radius - cherenkov_to_support_distance - support_dimensions[0] / 2., self.cherenkov.z - self.cherenkov.length / 2. + cherenkov_end_to_support_edge_distance + support_dimensions[1] / 2., self.cherenkov.theta))

        # NOvA DocDB 36405
        vinyl_window_thickness = 0.006 * Beamline.INCH
        tedlar_window_thickness = 0.00175 * Beamline.INCH
        self.f_out.write('tubs tedlar_window radius={} length={} color=0.74,0.34,0.09 material=POLYVINYL_CHLORIDE\n'.format(cherenkov_outer_radius, tedlar_window_thickness))
        self.f_out.write('tubs vinyl_window radius={} length={} color=0.74,0.34,0.09 material=POLYVINYL_CHLORIDE\n'.format(cherenkov_outer_radius, vinyl_window_thickness))
        self.f_out.write('place tedlar_window rename=tedlar_window_up x={} y={} z={}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z - self.cherenkov.length / 2. - tedlar_window_thickness / 2.))
        self.f_out.write('place tedlar_window rename=tedlar_window_down x={} y={} z={}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z + self.cherenkov.length / 2. + tedlar_window_thickness / 2.))
        self.f_out.write('place vinyl_window rename=vinyl_window_up x={} y={} z={}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z - self.cherenkov.length / 2. - tedlar_window_thickness - vinyl_window_thickness / 2.))
        self.f_out.write('place vinyl_window rename=vinyl_window_down x={} y={} z={}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z + self.cherenkov.length / 2. + tedlar_window_thickness + vinyl_window_thickness / 2.))

    def write_nova_plane(self):
        self.nova.theta = self.us_theta + self.ds_theta
        self.nova.length = 1.
        # alignment
        self.nova.z = 14617.377049180326
        self.nova.x = -1374.4731770833332
        self.nova.y = 76.1231770833333
        self.nova.height = 2531.9463541666664 * 1.2
        self.nova.width = self.nova.height
        rotation_y = -0.3492962692141463
        rotation_z = 0.8032983167050841

        self.f_out.write('{} nova height={} length={} width={} material=POLYVINYL_CHLORIDE color=0.8,0.8,0.8\n'.format(self.detector_type, self.nova.height, self.nova.length, self.nova.width))
        # self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y{}\n'.format(self.nova.x, self.nova.y, self.nova.z, self.nova.theta))
        self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y{},z{}\n'.format(self.nova.x, self.nova.y, self.nova.z, rotation_y, rotation_z))

    def write_nova(self):
        self.nova.theta = self.us_theta + self.ds_theta
        self.nova.length = 3900.
        self.f_out.write('{} nova height={} length={} width={} material=POLYVINYL_CHLORIDE color=0.39,0.39,0.39\n'.format(self.detector_type, self.nova.height, self.nova.length, self.nova.width))
        self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y{}\n'.format(self.nova.x, self.nova.y, self.nova.z + self.nova.length / 2., self.nova.theta))

    def write_shielding_block(self):
        steel_dimensions = [24. * Beamline.INCH, 16. * Beamline.INCH, 24. * Beamline.INCH]
        cement_dimensions = [24. * Beamline.INCH, 16. * Beamline.INCH, 24. * Beamline.INCH]
        concrete_bottom_dimensions = [12. * Beamline.INCH, 32. * Beamline.INCH, 24. * Beamline.INCH]
        concrete_top_dimensions = [4. * Beamline.INCH, 32. * Beamline.INCH, 24. * Beamline.INCH]

        self.f_out.write('group shielding_block\n')
        self.f_out.write('  box shielding_block_steel height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(steel_dimensions[0], steel_dimensions[1], steel_dimensions[2], self.kill))
        self.f_out.write('  box shielding_block_cement height={} length={} width={} material=CONCRETE color=0,1,1 kill={}\n'.format(cement_dimensions[0], cement_dimensions[1], cement_dimensions[2], self.kill))
        self.f_out.write('  box shielding_block_concrete_bottom height={} length={} width={} material=CONCRETE color=0,1,1 kill={}\n'.format(concrete_bottom_dimensions[0], concrete_bottom_dimensions[1], concrete_bottom_dimensions[2], self.kill))
        self.f_out.write('  box shielding_block_concrete_top height={} length={} width={} material=CONCRETE color=0,1,1 kill={}\n'.format(concrete_top_dimensions[0], concrete_top_dimensions[1], concrete_top_dimensions[2], self.kill))

        z_shift = concrete_top_dimensions[1] / 2. # shift in z by half of the full length in z to avoid geometry conflict in the group
        self.f_out.write('  place shielding_block_steel x={} y={} z={}\n'.format(0., 0., -steel_dimensions[1] / 2. + z_shift))
        self.f_out.write('  place shielding_block_cement x={} y={} z={}\n'.format(0., 0., cement_dimensions[1] / 2. + z_shift))
        self.f_out.write('  place shielding_block_concrete_bottom x={} y={} z={}\n'.format(0., -cement_dimensions[0] / 2. - concrete_bottom_dimensions[0] / 2., z_shift))
        self.f_out.write('  place shielding_block_concrete_top x={} y={} z={}\n'.format(0., cement_dimensions[0] / 2. + concrete_top_dimensions[0] / 2., z_shift))
        self.f_out.write('endgroup\n')

        self.shielding_block.height = concrete_top_dimensions[0] + steel_dimensions[0] + concrete_bottom_dimensions[0]
        self.shielding_block.length = concrete_top_dimensions[1]
        self.shielding_block.width = concrete_top_dimensions[2]

        # convert geometric center to model origin
        y_shift = 4. * Beamline.INCH

        self.f_out.write('place shielding_block rename=shielding_block_1 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_1.x, self.shielding_block_1.y + y_shift, self.shielding_block_1.z, self.shielding_block_1.theta))
        self.f_out.write('place shielding_block rename=shielding_block_2 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_2.x, self.shielding_block_2.y + y_shift, self.shielding_block_2.z, self.shielding_block_2.theta))
        self.f_out.write('place shielding_block rename=shielding_block_3 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_3.x, self.shielding_block_3.y + y_shift, self.shielding_block_3.z, self.shielding_block_3.theta))

    def write_housing(self):
        thickness = 10.
        radius = 3000.
        shift = 1000.
        length = 20000. + shift

        self.f_out.write('virtualdetector wall innerRadius={} radius={} length={} color=1,1,1\n'.format(radius, radius + thickness, length))
        self.f_out.write('virtualdetector cap innerRadius={} radius={} length={} color=1,1,1\n'.format(0, radius + thickness, thickness))
        self.f_out.write('place wall x={} y={} z={}\n'.format(0, 0, length / 2. - shift))
        self.f_out.write('place cap rename=cap_start x={} y={} z={}\n'.format(0, 0, -shift - thickness / 2.))
        self.f_out.write('place cap rename=cap_end x={} y={} z={}\n'.format(0, 0, length - shift + thickness / 2.))

    def write(self):
        self.read_alignment_data_beamline()
        self.read_alignment_data_beamline_helium_pipe()

        self.f_out.write('physics QGSP_BIC\n')
        self.f_out.write('param worldMaterial=Air\n')
        self.f_out.write('param histoFile=beam.root\n')

        self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector 0 1 0"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/zoom 1.5"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        if self.screen_shot:
            self.f_out.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')

        self.f_out.write('beam gaussian particle=$particle firstEvent=$first lastEvent=$last sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=$momentum\n')
        self.f_out.write('trackcuts kineticEnergyCut=20 keep=gamma,pi0,pi+,pi-,kaon+,kaon-,mu+,mu-,e+,e-,proton,anti_proton\n')

        self.write_target()
        self.write_collimator_us_alignment()
        # if not self.screen_shot:
        #     self.write_virtual_disk()
        self.write_wc()
        self.write_helium_pipe()
        self.write_magnet()
        self.write_collimator_ds()
        self.write_tof()
        self.write_cherenkov()
        self.write_nova_plane()
        self.write_shielding_block()

        print('finished write()')
        print('file written to {}'.format(self.g4bl_filename))

    def write_radiation_dosage(self):
        self.f_out.write('physics QGSP_BIC\n')
        self.f_out.write('param worldMaterial=Air\n')
        self.f_out.write('param histoFile=beam.root\n')

        self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector -1 1 1"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/zoom 1.5"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        if self.screen_shot:
            self.f_out.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')

        self.f_out.write('beam gaussian particle=$particle firstEvent=$first lastEvent=$last sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=$momentum\n')
        self.f_out.write('trackcuts keep=pi+,pi-,pi0,kaon+,kaon-,mu+,mu-,proton,anti_proton,neutron,anti_neutron\n')

        self.write_target()
        self.write_collimator_us()
        if not self.screen_shot:
            self.write_virtual_disk()
        self.write_wc()
        self.write_magnet()
        self.write_collimator_ds()
        self.write_tof()
        self.write_cherenkov()
        self.write_shielding_block()
        self.write_nova()
        self.write_housing()

        print('finished write_radiation_dosage()')
        print('file written to {}'.format(self.g4bl_filename))

    def write_radiation_collimator(self):
        self.kill = 1

        self.f_out.write('physics QGSP_BIC\n')
        self.f_out.write('param worldMaterial=Air\n')
        self.f_out.write('param histoFile=beam.root\n')

        # self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector 0 1 0"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector -1 1 1"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/zoom 1.5"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        # self.f_out.write('g4ui when=4 "/vis/scene/add/axes 0 0 0 2 m"\n')
        self.f_out.write('g4ui when=4 "/vis/scene/add/axes 0 0 0 0.5 m"\n')
        if self.screen_shot:
            self.f_out.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')

        self.f_out.write('beam gaussian particle=$particle firstEvent=$first lastEvent=$last sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=$momentum\n')
        self.f_out.write('trackcuts keep=pi+,pi-,pi0,kaon+,kaon-,mu+,mu-,e+,e-,gamma,proton,anti_proton,neutron,anti_neutron\n')

        self.write_target()
        self.write_collimator_us()

        det_width = (6. + 1.) * Beamline.INCH
        det_height = (2. + 1.) * Beamline.INCH
        det_length = 1.
        det_r = 1440.

        det_positions = [det_r * sin(self.us_theta * Beamline.RADIAN_PER_DEGREE), 0., det_r * cos(self.us_theta * Beamline.RADIAN_PER_DEGREE)]
        self.f_out.write('virtualdetector det width={} height={} length={} material=Air color=0.9,0.9,0.7\n'.format(det_width, det_height, det_length))
        self.f_out.write('place det rotation=y{} x={} y={} z={}\n'.format(self.us_theta, det_positions[0], det_positions[1], det_positions[2]))

        print('finished write_radiation_collimator()')
        print('file written to {}'.format(self.g4bl_filename))

    def write_geometry_check(self):
        self.screen_shot = True
        self.kill = 1
        self.f_out.write('physics QGSP_BIC\n')
        self.f_out.write('param worldMaterial=Air\n')
        self.f_out.write('param histoFile=beam.root\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector -1 1 1"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/zoom 1.5"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        self.f_out.write('g4ui when=4 "/vis/scene/add/axes 0 0 0 0.5 m"\n')
        if self.screen_shot:
            self.f_out.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')
        self.f_out.write('beam gaussian particle=$particle firstEvent=$first lastEvent=$last sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=$momentum\n')
        self.f_out.write('trackcuts keep=pi+,pi-,pi0,kaon+,kaon-,mu+,mu-,e+,e-,gamma,proton,anti_proton,neutron,anti_neutron\n')

        # upstream collimator
        # self.write_target()
        # self.write_collimator_us()
        # self.write_collimator_us_alignment()

        # det_width = (6. + 1.) * Beamline.INCH
        # det_height = (2. + 1.) * Beamline.INCH
        # det_length = 1.
        # det_r = 1440.

        # # check upstream opening width
        # # det_width = 2.83 * Beamline.INCH
        # # det_r = 210.
        # self.us_theta = 0.
        # det_width = 6 * Beamline.INCH
        # det_positions = [-376, 0., 1300]

        # # check downstream opening width
        # # det_width = 6 * Beamline.INCH
        # # det_r = 1440.
        # self.us_theta = 0.
        # det_width = 6 * Beamline.INCH
        # det_positions = [-376, 0., 1300]

        # # check downstream opening to the left edge
        # self.us_theta = 0.
        # det_width = 6.5 * Beamline.INCH
        # det_positions = [-541, 0., 1300]

        # # check downstream opening to the right edge
        # self.us_theta = 0.
        # det_width = 19.5 * Beamline.INCH
        # det_positions = [-58, 0., 1350]

        # det_positions = [det_r * sin(self.us_theta * Beamline.RADIAN_PER_DEGREE), 0., det_r * cos(self.us_theta * Beamline.RADIAN_PER_DEGREE)]
        # self.f_out.write('virtualdetector det width={} height={} length={} material=Air color=0.9,0.9,0.7\n'.format(det_width, det_height, det_length))
        # self.f_out.write('place det rotation=y{} x={} y={} z={}\n'.format(self.us_theta, det_positions[0], det_positions[1], det_positions[2]))

        # downstream collimator
        # self.collimator_ds.x = 0.
        # self.collimator_ds.y = 0.
        # self.collimator_ds.z = 0.
        # self.write_collimator_ds()

        # tofs
        # self.tof_us.x = 0.
        # self.tof_us.y = 0.
        # self.tof_us.z = 0.
        # self.tof_us.theta = 0.
        # self.write_tof()

        # Cherenkov counter
        # self.cherenkov.x = 0.
        # self.cherenkov.y = 0.
        # self.cherenkov.z = 0.
        # self.write_cherenkov()

        # magnet
        # self.magnet.x = 0.
        # self.magnet.y = 0.
        # self.magnet.z = 0.
        # self.magnet.theta = 0.
        # self.write_magnet()

        # shielding block
        # self.shielding_block_1.x = 0.
        # self.shielding_block_1.y = 0.
        # self.shielding_block_1.z = 0.
        # self.shielding_block_1.theta = 0.
        # self.write_shielding_block()

        # wire chambers
        self.wc_1.set_xyz([0., 0., 0.])
        self.write_wc()

        # nova detector
        # # self.nova.theta = self.us_theta + self.ds_theta
        # self.nova.length = 10.
        # # self.nova.z = 14617.377049180326
        # # self.nova.x = -1374.4731770833332
        # # self.nova.y = 76.1231770833333
        # self.nova.height = 2531.9463541666664 * 1.2
        # self.nova.width = self.nova.height
        # self.nova.z = 0.
        # self.nova.x = 0.
        # self.nova.y = 0.
        # # self.nova.height = 10.
        # # self.nova.width = 1000.
        # self.f_out.write('virtualdetector nova height={} length={} width={} color=0.39,0.39,0.39\n'.format(self.nova.height, self.nova.length, self.nova.width))
        # # self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y{}\n'.format(self.nova.x, self.nova.y, self.nova.z, self.nova.theta))
        # # self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y0,z0\n'.format(self.nova.x, self.nova.y, self.nova.z))
        # # self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y45,z0\n'.format(self.nova.x, self.nova.y, self.nova.z))
        # # self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y45,z-90\n'.format(self.nova.x, self.nova.y, self.nova.z))
        # self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y-0.35,z0.8\n'.format(self.nova.x, self.nova.y, self.nova.z))


        # helium pipes
        # outer_radius = 6. * Beamline.INCH / 2.
        # wall_thickness = 3. / 32. * Beamline.INCH
        # inner_radius = outer_radius - wall_thickness
        # mylar_window_thickness = 0.003 * Beamline.INCH

        # self.helium_pipe_1.theta = 0.
        # self.helium_pipe_1.x = 0.
        # self.helium_pipe_1.y = 0.
        # self.helium_pipe_1.z = 0.

        # self.f_out.write('tubs mylar_window radius={} length={} color=0,0,1 material=MYLAR kill=0\n'.format(outer_radius, mylar_window_thickness))
        # for i, helium_pipe in enumerate([self.helium_pipe_1]):
        #     index = i + 1
        #     self.f_out.write('group helium_pipe_{}\n'.format(index))
        #     self.f_out.write('  tubs helium radius={} length={} color=1,1,1 material=He kill=0\n'.format(inner_radius, helium_pipe.length))
        #     self.f_out.write('  tubs helium_pipe innerRadius={} outerRadius={} length={} color=0,0.8,0 material=STAINLESS-STEEL kill={}\n'.format(inner_radius, outer_radius, helium_pipe.length, self.kill))

        #     z_shift = (helium_pipe.length + mylar_window_thickness * 2.) / 2. # shift in z by half of the full length in z to avoid geometry conflict in the group
        #     self.f_out.write('  place helium_pipe rename=helium_pipe x={} y={} z={}\n'.format(0., 0., z_shift))
        #     self.f_out.write('  place helium rename=helium x={} y={} z={}\n'.format(0., 0., z_shift))
        #     self.f_out.write('  place mylar_window rename=mylar_window_up x={} y={} z={}\n'.format(0., 0., -helium_pipe.length / 2. - mylar_window_thickness / 2. + z_shift))
        #     self.f_out.write('  place mylar_window rename=mylar_window_down x={} y={} z={}\n'.format(0., 0., helium_pipe.length / 2. + mylar_window_thickness / 2. + z_shift))
        #     self.f_out.write('endgroup\n')
        #     self.f_out.write('place helium_pipe_{} rename=helium_pipe_{} x={} y={} z={} rotation=y{}\n'.format(index, index, helium_pipe.x, helium_pipe.y, helium_pipe.z, helium_pipe.theta))

        print('finished write_geometry_check()')
        print('file written to {}'.format(self.g4bl_filename))

    def plot_alignment_data(self, name_positions, names, figure_name, **kwargs):
        legend_text_size = kwargs.get('legend_text_size', None)
        legend_n_columns = kwargs.get('legend_n_columns', None)

        print('len(names) = {}'.format(len(names)))
        pprint(name_positions)

        poly_mks = []
        mk_zxs = {}
        mk_zys = {}
        xs = []
        ys = []
        zs = []
        for i, name in enumerate(names):
            x = -name_positions[name][0]
            y = name_positions[name][2]
            z = name_positions[name][1]

            mk_zxs[name] = TMarker(z, x, 20)
            mk_zys[name] = TMarker(z, y, 20)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            poly_mk = TPolyMarker3D(1)
            poly_mk.SetPoint(0, z, x, y)
            poly_mks.append(poly_mk)

        gr_zx = TGraph(len(names), np.array(zs), np.array(xs))
        gr_zy = TGraph(len(names), np.array(zs), np.array(ys))
        gr_2d = TGraph2D(len(names), np.array(zs), np.array(xs), np.array(ys))

        for gr in [gr_zx, gr_zy]:
            set_graph_style(gr)
            gr.SetMarkerSize(0)
            gr.SetMarkerColor(kWhite)
            gr.GetXaxis().SetTitle('Z (in)')
            gr.GetXaxis().SetTitleOffset(2.5)
            gr.GetYaxis().SetTitleOffset(2)

        c1 = TCanvas('c1', 'c1', 1600, 1000)
        c1.Divide(2, 2)

        c1.cd(1)
        set_margin()
        gr_2d.Draw('P')
        set_graph_style(gr_2d)
        gr_2d.SetMarkerSize(0)
        gr_2d.SetMarkerColor(kWhite)
        gr_2d.GetXaxis().SetTitle('Z (in)')
        gr_2d.GetYaxis().SetTitle('X (in)')
        gr_2d.GetZaxis().SetTitle('Y (in)')
        gr_2d.GetXaxis().SetTitleOffset(3)
        gr_2d.GetYaxis().SetTitleOffset(4)
        gr_2d.GetZaxis().SetTitleOffset(2)

        for i, poly_mk in enumerate(poly_mks):
            poly_mk.SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            poly_mk.SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])
            poly_mk.SetMarkerSize(2)
            poly_mk.Draw()

        c1.cd(2)
        set_margin()
        gr_zx.Draw('AP')
        gr_zx.GetYaxis().SetTitle('X (in)')
        lg1 = TLegend(0.1, 0, 0.9, 0.9)
        set_legend_style(lg1)
        if legend_n_columns:
            lg1.SetNColumns(legend_n_columns)
        if legend_text_size:
            lg1.SetTextSize(legend_text_size)

        for i, name in enumerate(names):
            mk_zx = mk_zxs[name]
            mk_zx.Draw()
            mk_zx.SetMarkerSize(2)
            mk_zx.SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            mk_zx.SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])
            lg1.AddEntry(mk_zx, name, 'p')

        c1.cd(3)
        set_margin()
        gr_zy.Draw('AP')
        gr_zy.GetYaxis().SetTitle('Y (in)')
        for i, name in enumerate(names):
            mk_zy = mk_zys[name]
            mk_zy.Draw()
            mk_zy.SetMarkerSize(2)
            mk_zy.SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            mk_zy.SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])

        c1.cd(4)
        lg1.Draw()

        c1.Update()
        c1.SaveAs('{}/{}.pdf'.format(self.figure_dir, figure_name))
        input('Press any key to continue.')

    def read_alignment_data_beamline(self):
        # for component in self.components:
        #     component.set_xyz((0., 0., 0.))

        with open('alignment/NTB summary_up_ct_dn.txt') as f_txt:
            for row in csv.reader(f_txt, delimiter=','):
                detector_name = row[1].strip()
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                position = -x, z, y

                if '_CT' in detector_name:
                    if 'TARGET' in detector_name:
                        self.target.set_xyz(position)
                    elif 'NTB-TGT-COLL-002-TOF-1' in detector_name:
                        self.tof_us.set_xyz(position)
                    elif 'NTB-MWPC-AK' in detector_name:
                        self.wc_1.set_xyz(position)
                    elif 'NTB-MWPC-AL' in detector_name:
                        self.wc_2.set_xyz(position)
                    elif 'NTB-MWPC-AF' in detector_name:
                        self.wc_3.set_xyz(position)
                    elif 'NTB-MWPC-AI' in detector_name:
                        self.wc_4.set_xyz(position)
                    elif 'NTB-CERENKOV-TOF-1' in detector_name:
                        self.tof_ds.set_xyz(position)
                    elif 'NTB-CERENKOV-TOF-2' in detector_name:
                        self.tof_ds_sipm.set_xyz(position)
                    elif 'NTB-COLLIMATOR' in detector_name:
                        self.collimator_ds.set_xyz(position)
                    elif 'NTB-M-1-0' in detector_name:
                        self.magnet.set_xyz(position)
                    elif 'NTB-MIPP-SHIELD-BLOCK' in detector_name:
                        self.shielding_block_1.set_xyz(position)
                    elif 'NTB-LEFT-SHIELD-BLOCK' in detector_name:
                        self.shielding_block_2.set_xyz(position)
                    elif 'NTB-RIGHT-SHIELD-BLOCK' in detector_name:
                        self.shielding_block_3.set_xyz(position)

        for component in self.components:
            if component.name not in ['nova detector', 'upstream collimator']:
                component.x *= 25.4
                component.y *= 25.4
                component.z *= 25.4

    def read_alignment_data_beamline_helium_pipe(self, **kwargs):
        name_positions = {}

        with open('alignment/NTB summary_up_ct_dn.txt') as f_txt:
            for row in csv.reader(f_txt, delimiter=','):
                detector_name = row[1].strip()
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                position = -x, z, y

                if 'BEAMPIPE' in detector_name:
                    name = detector_name[:-3]
                    if name not in name_positions:
                        name_positions[name] = []
                    name_positions[name].append(position)

        name_lengths = {}
        for name in name_positions:
            length = 0.
            for i in range(len(name_positions[name][0])):
                length += (name_positions[name][0][i] - name_positions[name][1][i])**2
            length = length**0.5 * Beamline.INCH
            name_lengths[name] = length

        name_centers = {}
        for name in name_positions:
            name_centers[name] = []
            for i in range(len(name_positions[name][0])):
                name_centers[name].append((name_positions[name][0][i] + name_positions[name][1][i]) / 2. * Beamline.INCH)

        for name in name_centers:
            xyz_length = (name_centers[name][0], name_centers[name][1], name_centers[name][2], name_lengths[name])
            if name == 'NTB-MWPC-AK-BEAMPIPE':
                self.helium_pipe_1.x, self.helium_pipe_1.y, self.helium_pipe_1.z, self.helium_pipe_1.length = xyz_length
            elif name == 'NTB-M-1-0-BEAMPIPE':
                self.helium_pipe_2.x, self.helium_pipe_2.y, self.helium_pipe_2.z, self.helium_pipe_2.length = xyz_length
            elif name == 'NTB-MWPC-AF-BEAMPIPE':
                self.helium_pipe_3.x, self.helium_pipe_3.y, self.helium_pipe_3.z, self.helium_pipe_3.length = xyz_length
            elif name == 'NTB-MWPC-AI-BEAMPIPE':
                self.helium_pipe_4.x, self.helium_pipe_4.y, self.helium_pipe_4.z, self.helium_pipe_4.length = xyz_length

    def read_alignment_data_beamline_collimator_us(self, **kwargs):
        plot = kwargs.get('plot', True)

        mwpc_txt_name_positions = {
            'NTB-MWPC-P_UP': (0.010204, -26.058411, -0.034998),
            'NTB-MWPC-P_CT': (0.015523, -25.153253, -0.009607),
            'NTB-MWPC-P_DN': (0.020869, -24.248055, 0.015647)
        }

        mwpc_pdf_name_positions = {
            'NTB-MWPC-P_ROLL': (-0.333, -27.957, 99.951),
            'NTB-MWPC-P_UP': (0.010, -26.058, -0.035),
            'NTB-MWPC-P_CT': (0.016, -25.153, -0.010),
            'NTB-MWPC-P_DN': (0.021, -24.248, 0.016),
            'NTB-MWPC-P-0.875_A': (-4.093, -26.572, -0.036),
            'NTB-MWPC-P-0.875_B': (-0.002, -26.705, 3.997),
            'NTB-MWPC-P-0.875_C': (3.908, -26.633, 0.271),
            'NTB-MWPC-P-0.875_D': (-0.113, -26.494, -3.703),
            'NTB-MWPC-P-0.875_E': (-3.500, -23.702, 0.147),
            'NTB-MWPC-P-0.875_F': (0.241, -23.621, -3.645),
            'NTB-MWPC-P-0.875_G': (3.673, -23.736, 0.046),
            'NTB-MWPC-P-0.875_H': (0.424, -23.814, 3.627)
        }

        collimator_txt_name_positions = {
            'NTB-TGT-COLL-002_UP': (2.214573, 7.693723, -0.039740),
            'NTB-TGT-COLL-002-CHANNEL_UP': (2.214573, 7.693723, -0.039740),
            'NTB-TGT-COLL-002_DN': (2.220929, 50.429730, 0.081151),
            'NTB-TGT-COLL-002-CHANNEL_DN': (14.561032, 50.438895, 0.012260),
            'NTB-TARGET_UP': (0.012334,  -6.282132,  -0.165835),
            'NTB-TARGET_CT': (0.015489,  -0.000110,  -0.141833),
            'NTB-TARGET_DN': (0.018712,  6.281790,  -0.117855)
        }

        collimator_pdf_name_positions = {
            'NTB-TGT-COLL-002_ROLL': (2.563, 7.471, 99.960),
            'NTB-TGT-COLL-002_UP': (2.215, 7.694, -0.040),
            'NTB-TGT-COLL-002-CHANNEL_UP': (2.215, 7.694, -0.040),
            'NTB-TGT-COLL-002_DN': (2.221, 50.430, 0.081),
            'NTB-TGT-COLL-002-CHANNEL_DN': (14.561, 50.439, 0.012),
            'NTB-TGT-COLL-002_C': (24.779, -4.807, -9.798),
            'NTB-TGT-COLL-002_D': (21.051, -8.712, -9.819),
            'NTB-TGT-COLL-002_E': (-5.761, -8.724, -9.843),
            'NTB-TGT-COLL-002_F': (-9.327, -5.378, -9.510),
            'NTB-TGT-COLL-002_L': (24.821, 47.911, -9.740),
            'NTB-TGT-COLL-002_M': (20.929, 51.553, -9.693),
            'NTB-TGT-COLL-002_N': (-5.753, 51.529, -9.394),
            'NTB-TGT-COLL-002_P': (-9.375, 47.952, -9.525),
            'NTB-TARGET_UP': (0.012, -6.282, -0.166),
            'NTB-TARGET_CT': (0.015, 0.000, -0.142),
            'NTB-TARGET_DN': (0.019, 6.282, -0.118),
            'NTB-TARGET_ROLL': (-1.352, -0.382, 99.848)
        }

        mwpc_txt_names = sorted(mwpc_txt_name_positions.keys())
        mwpc_pdf_names = sorted(mwpc_pdf_name_positions.keys())
        collimator_txt_names = sorted(collimator_txt_name_positions.keys())
        collimator_pdf_names = [
            'NTB-TARGET_UP',
            'NTB-TARGET_CT',
            'NTB-TARGET_DN',
            # 'NTB-TARGET_ROLL',
            'NTB-TGT-COLL-002_UP',
            'NTB-TGT-COLL-002_DN',
            # 'NTB-TGT-COLL-002_ROLL',
            'NTB-TGT-COLL-002-CHANNEL_UP',
            'NTB-TGT-COLL-002-CHANNEL_DN',
            'NTB-TGT-COLL-002_C',
            'NTB-TGT-COLL-002_D',
            'NTB-TGT-COLL-002_E',
            'NTB-TGT-COLL-002_F',
            'NTB-TGT-COLL-002_L',
            'NTB-TGT-COLL-002_M',
            'NTB-TGT-COLL-002_N',
            'NTB-TGT-COLL-002_P',
        ]

        if plot:
            self.plot_alignment_data(collimator_pdf_name_positions, collimator_pdf_names, 'read_alignment_data_beamline_collimator_us.plot')

        return collimator_pdf_name_positions['NTB-TGT-COLL-002-CHANNEL_UP'], collimator_pdf_name_positions['NTB-TGT-COLL-002-CHANNEL_DN']

    def read_alignment_data_beamline_mwpc(self, **kwargs):
        plot = kwargs.get('plot', True)

        mwpc_1_pdf_name_positions = {
            'NTB-MWPC-AK-0.875_C': (20.665, 60.294, 0.060),
            'NTB-MWPC-AK-BOT-BR-PIN_UP': (21.237, 60.625, 3.770),
            'NTB-MWPC-AK-BOT-BL-PIN_UP': (21.274, 60.700, -3.728),
            'NTB-MWPC-AK-0.875_D': (17.652, 61.256, -3.346),
            'NTB-MWPC-AK-0.875_B': (17.296, 61.287, 3.342),
            'NTB-MWPC-AK_UP': (17.667, 61.761, 0.011),
            'NTB-MWPC-AK-0.875_A': (14.331, 62.230, 0.176),
            'NTB-MWPC-AK-BOT-BR-PIN_DN': (21.779, 62.379, 3.817),
            'NTB-MWPC-AK-BOT-BL-PIN_DN': (21.814, 62.456, -3.686),
            'NTB-MWPC-AK_CT': (17.943, 62.638, 0.021),
            'NTB-MWPC-AK-TOP-BR-PIN_UP': (14.055, 62.823, 3.755),
            'NTB-MWPC-AK-TOP-BL-PIN_UP': (14.099, 62.896, -3.747),
            'NTB-MWPC-AK-0.875_G': (21.571, 63.033, -0.056),
            'NTB-MWPC-AK_DN': (18.218, 63.516, 0.031),
            'NTB-MWPC-AK-0.875_H': (18.330, 63.988, 3.399),
            'NTB-MWPC-AK-0.875_F': (18.508, 64.011, -3.272),
            'NTB-MWPC-AK-TOP-BR-PIN_DN': (14.611, 64.579, 3.805),
            'NTB-MWPC-AK-TOP-BL-PIN_DN': (14.646, 64.656, -3.698),
            'NTB-MWPC-AK-0.875_E': (15.175, 64.996, 0.027),
            'NTB-MWPC-AK_ROLL': (-77.470, 92.578, -0.179),
        }
        mwpc_1_pdf_names = [
            'NTB-MWPC-AK-BOT-BR-PIN_UP',
            'NTB-MWPC-AK-BOT-BL-PIN_UP',
            'NTB-MWPC-AK-BOT-BR-PIN_DN',
            'NTB-MWPC-AK-BOT-BL-PIN_DN',
            'NTB-MWPC-AK-TOP-BR-PIN_UP',
            'NTB-MWPC-AK-TOP-BL-PIN_UP',
            'NTB-MWPC-AK-TOP-BR-PIN_DN',
            'NTB-MWPC-AK-TOP-BL-PIN_DN',
            'NTB-MWPC-AK-0.875_A',
            'NTB-MWPC-AK-0.875_B',
            'NTB-MWPC-AK-0.875_C',
            'NTB-MWPC-AK-0.875_D',
            'NTB-MWPC-AK-0.875_E',
            'NTB-MWPC-AK-0.875_F',
            'NTB-MWPC-AK-0.875_G',
            'NTB-MWPC-AK-0.875_H',
            'NTB-MWPC-AK_UP',
            'NTB-MWPC-AK_CT',
            'NTB-MWPC-AK_DN',
            # 'NTB-MWPC-AK_ROLL',
        ]

        mwpc_2_pdf_name_positions = {
            'NTB-MWPC-AL-BOT-BL-PIN_UP': (37.191, 116.138, -3.723),
            'NTB-MWPC-AL-BOT-BR-PIN_UP': (37.238, 116.186, 3.775),
            'NTB-MWPC-AL-BOT-BL-PIN_DN': (37.700, 117.902, -3.739),
            'NTB-MWPC-AL-BOT-BR-PIN_DN': (37.738, 117.953, 3.766),
            'NTB-MWPC-AL-TOP-BL-PIN_UP': (29.972, 118.182, -3.691),
            'NTB-MWPC-AL-TOP-BR-PIN_UP': (30.019, 118.230, 3.810),
            'NTB-MWPC-AL-TOP-BL-PIN_DN': (30.480, 119.943, -3.715),
            'NTB-MWPC-AL-TOP-BR-PIN_DN': (30.519, 119.994, 3.788),
            'NTB-MWPC-AL-0.875_A': (30.289, 117.580, -0.049),
            'NTB-MWPC-AL-0.875_B': (33.457, 116.711, 3.370),
            'NTB-MWPC-AL-0.875_C': (36.685, 115.767, -0.192),
            'NTB-MWPC-AL-0.875_D': (33.438, 116.663, -3.350),
            'NTB-MWPC-AL-0.875_E': (31.066, 120.352, 0.082),
            'NTB-MWPC-AL-0.875_F': (34.154, 119.451, -3.410),
            'NTB-MWPC-AL-0.875_G': (37.471, 118.541, 0.020),
            'NTB-MWPC-AL-0.875_H': (34.081, 119.526, 3.355),
            'NTB-MWPC-AL_UP': (33.606, 117.184, 0.043),
            'NTB-MWPC-AL_CT': (33.858, 118.066, 0.035),
            'NTB-MWPC-AL_DN': (34.110, 118.948, 0.028),
            'NTB-MWPC-AL_ROLL': (-62.291, 145.548, 0.408),
        }
        mwpc_2_pdf_names = [
            'NTB-MWPC-AL-BOT-BL-PIN_UP',
            'NTB-MWPC-AL-BOT-BR-PIN_UP',
            'NTB-MWPC-AL-BOT-BL-PIN_DN',
            'NTB-MWPC-AL-BOT-BR-PIN_DN',
            'NTB-MWPC-AL-TOP-BL-PIN_UP',
            'NTB-MWPC-AL-TOP-BR-PIN_UP',
            'NTB-MWPC-AL-TOP-BL-PIN_DN',
            'NTB-MWPC-AL-TOP-BR-PIN_DN',
            'NTB-MWPC-AL-0.875_A',
            'NTB-MWPC-AL-0.875_B',
            'NTB-MWPC-AL-0.875_C',
            'NTB-MWPC-AL-0.875_D',
            'NTB-MWPC-AL-0.875_E',
            'NTB-MWPC-AL-0.875_F',
            'NTB-MWPC-AL-0.875_G',
            'NTB-MWPC-AL-0.875_H',
            'NTB-MWPC-AL_UP',
            'NTB-MWPC-AL_CT',
            'NTB-MWPC-AL_DN',
            # 'NTB-MWPC-AL_ROLL',
        ]

        mwpc_3_pdf_name_positions = {
            'NTB-MWPC-AF-0.875_B': (53.249, 288.663, 3.565),
            'NTB-MWPC-AF-0.875_C': (56.823, 288.605, -0.059),
            'NTB-MWPC-AF-0.875_D': (53.179, 288.711, -3.586),
            'NTB-MWPC-AF-0.875_E': (49.791, 291.663, 0.073),
            'NTB-MWPC-AF-0.875_F': (53.342, 291.561, 3.631),
            'NTB-MWPC-AF-0.875_G': (56.820, 291.500, -0.006),
            'NTB-MWPC-AF-0.875_H': (53.294, 291.612, -3.496),
            'NTB-MWPC-AF_UP': (53.241, 289.214, -0.004),
            'NTB-MWPC-AF_CT': (53.262, 290.136, 0.003),
            'NTB-MWPC-AF_DN': (53.283, 291.058, 0.010),
            'NTB-MWPC-AF_ROLL': (52.802, 289.411, 99.999),
        }
        mwpc_3_pdf_names = [
            'NTB-MWPC-AF-0.875_B',
            'NTB-MWPC-AF-0.875_C',
            'NTB-MWPC-AF-0.875_D',
            'NTB-MWPC-AF-0.875_E',
            'NTB-MWPC-AF-0.875_F',
            'NTB-MWPC-AF-0.875_G',
            'NTB-MWPC-AF-0.875_H',
            'NTB-MWPC-AF_UP',
            'NTB-MWPC-AF_CT',
            'NTB-MWPC-AF_DN',
            # 'NTB-MWPC-AF_ROLL',
        ]

        mwpc_4_pdf_name_positions = {
            'NTB-MWPC-AI_ROLL': (-46.661, 396.623, 0.236),
            'NTB-MWPC-AI-0.500_D': (50.957, 398.049, -3.689),
            'NTB-MWPC-AI-0.500_A': (50.802, 398.096, 3.884),
            'NTB-MWPC-AI-0.500_C': (55.778, 398.159, -3.661),
            'NTB-MWPC-AI-0.500_B': (55.722, 398.210, 3.870),
            'NTB-MWPC-AI_UP': (53.327, 398.399, 0.019),
            'NTB-MWPC-AI_CT': (53.302, 399.344, 0.012),
            'NTB-MWPC-AI_DN': (53.276, 400.288, 0.006),
            'NTB-MWPC-AI-0.500_K': (50.026, 400.525, -0.235),
            'NTB-MWPC-AI-0.500_L': (53.435, 400.582, -3.294),
            'NTB-MWPC-AI-0.500_J': (53.179, 400.616, 3.296),
            'NTB-MWPC-AI-0.500_M': (56.607, 400.675, 0.071),
            'NTB-MWPC-AI-0.500_F': (57.663, 400.717, 3.448),
        }
        mwpc_4_pdf_names = [
            'NTB-MWPC-AI-0.500_A',
            'NTB-MWPC-AI-0.500_B',
            'NTB-MWPC-AI-0.500_C',
            'NTB-MWPC-AI-0.500_D',
            'NTB-MWPC-AI-0.500_F',
            'NTB-MWPC-AI-0.500_J',
            'NTB-MWPC-AI-0.500_K',
            'NTB-MWPC-AI-0.500_L',
            'NTB-MWPC-AI-0.500_M',
            'NTB-MWPC-AI_UP',
            'NTB-MWPC-AI_CT',
            'NTB-MWPC-AI_DN',
            # 'NTB-MWPC-AI_ROLL',
        ]

        if plot:
            self.plot_alignment_data(mwpc_1_pdf_name_positions, mwpc_1_pdf_names, 'read_alignment_data_beamline_mwpc_1.plot')
            # self.plot_alignment_data(mwpc_2_pdf_name_positions, mwpc_2_pdf_names, 'read_alignment_data_beamline_mwpc_2.plot')
            # self.plot_alignment_data(mwpc_3_pdf_name_positions, mwpc_3_pdf_names, 'read_alignment_data_beamline_mwpc_3.plot')
            # self.plot_alignment_data(mwpc_4_pdf_name_positions, mwpc_4_pdf_names, 'read_alignment_data_beamline_mwpc_4.plot')

        return mwpc_1_pdf_name_positions['NTB-MWPC-AK_CT'], mwpc_2_pdf_name_positions['NTB-MWPC-AL_CT'], mwpc_3_pdf_name_positions['NTB-MWPC-AF_CT'], mwpc_4_pdf_name_positions['NTB-MWPC-AI_CT']

    def read_alignment_data_beamline_magnet(self, **kwargs):
        plot = kwargs.get('plot', True)

        magnet_pdf_name_positions = {
            'NTB-M-1-0_A': (64.010, 164.929, 16.162),
            'NTB-M-1-0_B': (73.313, 164.856, 4.061),
            'NTB-M-1-0_D': (65.274, 173.812, -16.160),
            'NTB-M-1-0_E': (34.082, 178.498, -16.162),
            'NTB-M-1-0_F': (23.485, 169.334, -3.893),
            'NTB-M-1-0_G': (23.394, 169.441, 3.908),
            'NTB-M-1-0_H': (33.061, 171.541, 16.164),
            'NTB-M-1-0_J': (69.345, 203.052, 16.143),
            'NTB-M-1-0_K': (78.673, 202.851, 3.861),
            'NTB-M-1-0_L': (78.735, 202.771, -3.813),
            'NTB-M-1-0_M': (69.386, 202.894, -16.190),
            'NTB-M-1-0_N': (38.116, 207.227, -16.206),
            'NTB-M-1-0_Q': (29.145, 209.927, 3.833),
            'NTB-M-1-0_P': (29.189, 209.786, -3.893),
            'NTB-M-1-0_R': (38.137, 207.425, 16.138),

            'NTB-M-1-0_UP': (48.158, 165.318, 0.013),
            'NTB-M-1-0_CT': (51.089, 186.145, -0.010),
            'NTB-M-1-0_DN': (54.020, 206.972, -0.034),
            'NTB-M-1-0_ROLL': (51.086, 186.256, 99.990),

            'NTB-M-1-0-SAGITTA_UP': (47.430, 165.420, 0.013),
            'NTB-M-1-0-SAGITTA_CT': (50.361, 186.247, -0.010),
            'NTB-M-1-0-SAGITTA_DN': (53.292, 207.074, -0.034),

            'NTB-M-1-0-TOP-OF-SPACER_UP': (48.158, 165.318, 0.576),
            'NTB-M-1-0-TOP-OF-SPACER_CT': (51.089, 186.145, 0.552),
            'NTB-M-1-0-TOP-OF-SPACER_DN': (54.020, 206.972, 0.529),
            'NTB-M-1-0-TOP-OF-SPACER_ROLL': (51.086, 186.257, 100.552),

            'NTB-M-1-0-MFS-53-X': (64.778, 184.085, 1.523),
            'NTB-M-1-0-MFS-53-Y': (65.019, 186.079, -0.469),
            'NTB-M-1-0-MFS-53-Z': (-34.414, 196.870, 0.337),

            'NTB-M-1-0-MFS-53-X-SENSOR': (64.770, 184.200, -0.333),
            'NTB-M-1-0-MFS-53-Y-SENSOR': (64.799, 184.089, -0.347),
            'NTB-M-1-0-MFS-53-Z-SENSOR': (64.809, 184.169, -0.475),

            'NTB-M-1-0-MFS-53-ORIGIN': (64.763, 184.095, -0.477),
            'NTB-M-1-0-MFS-53-0.500-SMR-1': (64.798, 185.004, 0.502),
            'NTB-M-1-0-MFS-53-0.500-SMR-2': (64.670, 184.026, 0.513),
            'NTB-M-1-0-MFS-53-0.500-SMR-3': (64.658, 184.059, -1.306),
            'NTB-M-1-0-MFS-53-0.500-SMR-4': (64.782, 185.027, -1.221),

            'NTB-M-1-0-MFS-54-X': (37.477, 187.965, -2.361),
            'NTB-M-1-0-MFS-54-Y': (37.718, 189.894, -0.305),
            'NTB-M-1-0-MFS-54-Z': (136.451, 173.962, 1.099),

            'NTB-M-1-0-MFS-54-X-SENSOR': (37.458, 188.035, -0.483),
            'NTB-M-1-0-MFS-54-Y-SENSOR': (37.402, 187.921, -0.489),
            'NTB-M-1-0-MFS-54-Z-SENSOR': (37.406, 187.996, -0.360),

            'NTB-M-1-0-MFS-54-ORIGIN': (37.440, 187.914, -0.361),
            'NTB-M-1-0-MFS-54-0.500-SMR-1': (37.659, 188.831, -1.306),
            'NTB-M-1-0-MFS-54-0.500-SMR-2': (37.527, 187.873, -1.362),
            'NTB-M-1-0-MFS-54-0.500-SMR-3': (37.491, 187.823, 0.476),
            'NTB-M-1-0-MFS-54-0.500-SMR-4': (37.647, 188.807, 0.436),
        }

        magnet_pdf_names = [
            'NTB-M-1-0_A',
            'NTB-M-1-0_B',
            'NTB-M-1-0_D',
            'NTB-M-1-0_E',
            'NTB-M-1-0_F',
            'NTB-M-1-0_G',
            'NTB-M-1-0_H',
            'NTB-M-1-0_J',
            'NTB-M-1-0_K',
            'NTB-M-1-0_L',
            'NTB-M-1-0_M',
            'NTB-M-1-0_N',
            'NTB-M-1-0_Q',
            'NTB-M-1-0_P',
            'NTB-M-1-0_R',

            'NTB-M-1-0_UP',
            'NTB-M-1-0_CT',
            'NTB-M-1-0_DN',
            # 'NTB-M-1-0_ROLL',

            'NTB-M-1-0-SAGITTA_UP',
            'NTB-M-1-0-SAGITTA_CT',
            'NTB-M-1-0-SAGITTA_DN',

            'NTB-M-1-0-TOP-OF-SPACER_UP',
            'NTB-M-1-0-TOP-OF-SPACER_CT',
            'NTB-M-1-0-TOP-OF-SPACER_DN',
            # 'NTB-M-1-0-TOP-OF-SPACER_ROLL',

            'NTB-M-1-0-MFS-53-X',
            'NTB-M-1-0-MFS-53-Y',
            'NTB-M-1-0-MFS-53-Z',

            'NTB-M-1-0-MFS-53-X-SENSOR',
            'NTB-M-1-0-MFS-53-Y-SENSOR',
            'NTB-M-1-0-MFS-53-Z-SENSOR',

            'NTB-M-1-0-MFS-53-ORIGIN',
            'NTB-M-1-0-MFS-53-0.500-SMR-1',
            'NTB-M-1-0-MFS-53-0.500-SMR-2',
            'NTB-M-1-0-MFS-53-0.500-SMR-3',
            'NTB-M-1-0-MFS-53-0.500-SMR-4',

            'NTB-M-1-0-MFS-54-X',
            'NTB-M-1-0-MFS-54-Y',
            'NTB-M-1-0-MFS-54-Z',

            'NTB-M-1-0-MFS-54-X-SENSOR',
            'NTB-M-1-0-MFS-54-Y-SENSOR',
            'NTB-M-1-0-MFS-54-Z-SENSOR',

            'NTB-M-1-0-MFS-54-ORIGIN',
            'NTB-M-1-0-MFS-54-0.500-SMR-1',
            'NTB-M-1-0-MFS-54-0.500-SMR-2',
            'NTB-M-1-0-MFS-54-0.500-SMR-3',
            'NTB-M-1-0-MFS-54-0.500-SMR-4',
        ]

        if plot:
            print('len(magnet_pdf_name_positions) = {}'.format(len(magnet_pdf_name_positions)))
            print('len(magnet_pdf_names) = {}'.format(len(magnet_pdf_names)))
            self.plot_alignment_data(magnet_pdf_name_positions, magnet_pdf_names, 'read_alignment_data_beamline_magnet.plot', legend_text_size=15, legend_n_columns=2)

        return magnet_pdf_name_positions['NTB-M-1-0_CT']

    def plot_vertical_positions(self):
        h1 = TH1D('h1', 'h1', 200, -100, 100)

        for component in self.components:
            h1.Fill(component.y)
            if component.y > 20.:
                print('component.name = {}, component.y = {}'.format(component.name, component.y))

        c1 = TCanvas('c1', 'c1', 800, 600)
        set_margin()
        set_h1_style(h1)

        h1.Draw()
        h1.GetXaxis().SetTitle('Vertical Position (mm)')
        h1.GetYaxis().SetTitle('Detector Count')
        h1.GetXaxis().SetRangeUser(-20, 50)
        c1.Update()
        draw_statbox(h1, x1=0.72)

        c1.Update()
        c1.SaveAs('{}/plot_vertical_positions.pdf'.format(self.figure_dir))
        input('Press any key to continue.')

    def plot_alignment_data_nova_detector(self):
        name_positions = {}
        with open('alignment/Block 1 and Block 2 offset points Target frame.txt') as f_txt:
            rows = csv.reader(f_txt, delimiter=',')
            row_count = 0
            for i in range(5):
                next(rows)
            for row in rows:
                row_count += 1
                name = row[0].strip()
                x = float(row[-3])
                y = float(row[-2])
                z = float(row[-1])
                position = -x, z, y
                if name not in name_positions:
                    name_positions[name] = []
                name_positions[name].append(position)
            print('row_count = {}'.format(row_count))

        names = sorted(name_positions.keys())
        names = [
            'Top of Blue Steel Baseplate Offset Points',
            # block 1
            'Block 1 Upper Beam Left Horizontal End Caps Offset Points', # 1
            'Block 1 Lower Beam Left Horizontal End Caps Offset Points', # 2
            'Block 1 Horizontal Panel Bottom Offset Points', # 3
            'Block 1 BL top shifted - 3"', # 4
            'Block 1 Beam Left Vertical Panel Offset Points', # 5
            'Block 1 Beam Right Vertical Panel Offset Points', # 6
            'Block 1 UPST Plane Offset Points', # 7
            'Block 1 Center Split points projected', # 8
            'Block 1 Panel 30 Downstream Plane Offset Points', # 9
            # block 2
            'Block 2 Upper Beam Left Horizontal End Caps Offset Points', # 10
            'Block 2 Lower Beam Left Horizontal End Caps Offset Points', # 11
            'Block 2 Horizontal Panel Bottom Offset Points', # 12
            'Block 2 Beam Left Vertical Panel Offset Points', # 13
            'Block 2 Beam Right Vertical Panel Offset Points', # 14
            'Block 2 DNST Plane Offset Points', # 15
            'Block 2 Center Split points projected', # 16
            'Block 2 Panel 32 Upstream Plane Offset Points' # 17
        ]

        name_poly_mk_3ds = {}
        name_poly_mk_zxs = {}
        name_poly_mk_zys = {}
        xs = []
        ys = []
        zs = []
        row_count = 0
        for i, name in enumerate(names):
            print('name = {}, i = {}'.format(name, i))
            print('len(name_positions[name]) = {}'.format(len(name_positions[name])))

            positions = name_positions[name]
            row_count += len(positions)

            poly_mk_3d = TPolyMarker3D(len(positions))
            poly_mk_zx = TPolyMarker(len(positions))
            poly_mk_zy = TPolyMarker(len(positions))
            for i, position in enumerate(positions):
                x = position[0] / 1000. # m
                y = position[1] / 1000. # m
                z = position[2] / 1000. # m

                poly_mk_3d.SetPoint(i, z, x, y)
                poly_mk_zx.SetPoint(i, z, x)
                poly_mk_zy.SetPoint(i, z, y)
                xs.append(x)
                ys.append(y)
                zs.append(z)
            name_poly_mk_3ds[name] = poly_mk_3d
            name_poly_mk_zxs[name] = poly_mk_zx
            name_poly_mk_zys[name] = poly_mk_zy

        print('row_count = {}'.format(row_count))
        gr_zx = TGraph(len(zs), np.array(zs), np.array(xs))
        gr_zy = TGraph(len(zs), np.array(zs), np.array(ys))
        gr_2d = TGraph2D(len(zs), np.array(zs), np.array(xs), np.array(ys))
        for gr in [gr_zx, gr_zy]:
            set_graph_style(gr)
            gr.SetMarkerSize(0)
            gr.SetMarkerColor(kWhite)
            gr.GetXaxis().SetTitle('Z (m)')
            gr.GetXaxis().SetTitleOffset(2.5)
            gr.GetYaxis().SetTitleOffset(2)

        c1 = TCanvas('c1', 'c1', 1600, 1000)
        set_margin()
        c1.Divide(2, 2)

        c1.cd(1)
        set_margin()
        gr_2d.Draw('P')
        set_graph_style(gr_2d)
        gr_2d.SetMarkerSize(0)
        gr_2d.SetMarkerColor(kWhite)
        gr_2d.GetXaxis().SetTitle('Z (m)')
        gr_2d.GetYaxis().SetTitle('X (m)')
        gr_2d.GetZaxis().SetTitle('Y (m)')
        gr_2d.GetXaxis().SetTitleOffset(3)
        gr_2d.GetYaxis().SetTitleOffset(4)
        gr_2d.GetZaxis().SetTitleOffset(2)

        for i, name in enumerate(names):
            print('poly_mk_3ds')
            print('i = {}'.format(i))
            print('name = {}'.format(name))

            poly_mk_3d = name_poly_mk_3ds[name]
            poly_mk_3d.SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            poly_mk_3d.SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])
            poly_mk_3d.SetMarkerSize(2)
            poly_mk_3d.Draw()

        c1.cd(2)
        set_margin()
        gr_zx.Draw('AP')
        gr_zx.GetYaxis().SetTitle('X (in)')

        lg1 = TLegend(0, 0, 0.4, 0.9)

        for i, name in enumerate(names):
            poly_mk_zx = name_poly_mk_zxs[name]
            poly_mk_zx.Draw()
            poly_mk_zx.SetMarkerSize(2)
            poly_mk_zx.SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            poly_mk_zx.SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])
            lg1.AddEntry(poly_mk_zx, name, 'p')

        c1.cd(3)
        set_margin()
        gr_zy.Draw('AP')
        gr_zy.GetYaxis().SetTitle('Y (in)')
        for i, name in enumerate(names):
            poly_mk_zy = name_poly_mk_zys[name]
            poly_mk_zy.Draw()
            poly_mk_zy.SetMarkerSize(2)
            poly_mk_zy.SetMarkerStyle(Beamline.MARKER_STYLES[i % len(Beamline.MARKER_STYLES)])
            poly_mk_zy.SetMarkerColor(Beamline.COLORS[i % len(Beamline.COLORS)])

        c1.cd(4)
        set_legend_style(lg1)
        lg1.SetTextSize(25)
        lg1.Draw()

        c1.Update()
        c1.SaveAs('{}/plot_alignment_data_nova_detector.pdf'.format(self.figure_dir))
        input('Press any key to continue.')

    def plot_alignment_data_nova_detector_edge(self):
        group_name_positions = {}
        with open('alignment/Block 1 and Block 2 offset points Target frame.txt') as f_txt:
            rows = csv.reader(f_txt, delimiter=',')
            for i in range(5):
                next(rows)
            for row in rows:
                group = row[0].strip()
                name = row[1].strip()
                x = float(row[-3])
                y = float(row[-2])
                z = float(row[-1])
                position = -x, z, y
                if group not in group_name_positions:
                    group_name_positions[group] = {}
                if name not in group_name_positions[group]:
                    group_name_positions[group][name] = (-x, z, y)

        block_horizontal_top_plane_positions = {}
        block_horizontal_bottom_plane_positions = {}
        block_vertical_left_plane_positions = {}
        block_vertical_right_plane_positions = {}
        module_widths = []

        block = 'block 1'
        group = 'Block 1 BL top shifted - 3"'
        block_horizontal_top_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-1])
            if plane not in block_horizontal_top_plane_positions[block]:
                block_horizontal_top_plane_positions[block][plane] = []
            block_horizontal_top_plane_positions[block][plane].append(group_name_positions[group][name])

        group = 'Block 1 Horizontal Panel Bottom Offset Points'
        block_horizontal_bottom_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-1])
            left_right = name.split(' ')[-4]
            # if 'BR' in left_right:
            #     continue
            if plane not in block_horizontal_bottom_plane_positions[block]:
                block_horizontal_bottom_plane_positions[block][plane] = []
            block_horizontal_bottom_plane_positions[block][plane].append(group_name_positions[group][name])

        group = 'Block 1 Beam Left Vertical Panel Offset Points'
        block_vertical_left_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-4])
            if plane not in block_vertical_left_plane_positions[block]:
                block_vertical_left_plane_positions[block][plane] = []
            block_vertical_left_plane_positions[block][plane].append(group_name_positions[group][name])

        group = 'Block 1 Beam Right Vertical Panel Offset Points'
        block_vertical_right_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-2])
            if plane not in block_vertical_right_plane_positions[block]:
                block_vertical_right_plane_positions[block][plane] = []
            block_vertical_right_plane_positions[block][plane].append(group_name_positions[group][name])

        block = 'block 2'
        group = 'Block 2 Horizontal Panel Bottom Offset Points'
        block_horizontal_bottom_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-5])
            left_right = name.split(' ')[-4]
            # if 'BR' in left_right:
            #     continue
            if plane not in block_horizontal_bottom_plane_positions[block]:
                block_horizontal_bottom_plane_positions[block][plane] = []
            block_horizontal_bottom_plane_positions[block][plane].append(group_name_positions[group][name])

        group = 'Block 2 Beam Left Vertical Panel Offset Points'
        block_vertical_left_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-4])
            if plane not in block_vertical_left_plane_positions[block]:
                block_vertical_left_plane_positions[block][plane] = []
            block_vertical_left_plane_positions[block][plane].append(group_name_positions[group][name])

        group = 'Block 2 Beam Right Vertical Panel Offset Points'
        block_vertical_right_plane_positions[block] = {}
        for name in group_name_positions[group]:
            plane = int(name.split(' ')[-4])
            if plane not in block_vertical_right_plane_positions[block]:
                block_vertical_right_plane_positions[block][plane] = []
            block_vertical_right_plane_positions[block][plane].append(group_name_positions[group][name])

        horizontal_planes = []
        horizontal_bottom_ys = []
        horizontal_bottom_y_up_to_plane_60s = []
        for block in block_horizontal_bottom_plane_positions:
            for plane in block_horizontal_bottom_plane_positions[block]:
                bottom_ys = [position[1] for position in block_horizontal_bottom_plane_positions[block][plane]]
                avg_bottom_y = sum(bottom_ys) / len(bottom_ys)
                horizontal_planes.append(float(plane))
                horizontal_bottom_ys.append(avg_bottom_y)
                if plane != 62:
                    horizontal_bottom_y_up_to_plane_60s.append(avg_bottom_y)

        vertical_planes = []
        vertical_left_xs = []
        for block in block_vertical_right_plane_positions:
            for plane in block_vertical_right_plane_positions[block]:
                left_xs = [position[0] for position in block_vertical_left_plane_positions[block][plane]]
                right_xs = [position[0] for position in block_vertical_right_plane_positions[block][plane]]
                avg_left_x = sum(left_xs) / len(left_xs)
                avg_right_x = sum(right_xs) / len(right_xs)
                module_width = avg_left_x - avg_right_x
                vertical_planes.append(float(plane))
                vertical_left_xs.append(avg_left_x)
                module_widths.append(module_width)

        module_width = sum(module_widths) / len(module_widths)
        # avg_horizontal_bottom_y = sum(horizontal_bottom_y_up_to_plane_60s) / len(horizontal_bottom_y_up_to_plane_60s)
        # print('avg_horizontal_bottom_y = {}'.format(avg_horizontal_bottom_y))
        # print('average vertical center = {}'.format(avg_horizontal_bottom_y + module_width / 2.))
        first_horizontal_bottom_ys = [position[1] for position in block_horizontal_bottom_plane_positions['block 1'][2]]
        first_horizontal_bottom_y = sum(first_horizontal_bottom_ys) / len(first_horizontal_bottom_ys)
        print('module_width = {}'.format(module_width))
        print('first_horizontal_bottom_y = {}'.format(first_horizontal_bottom_y))
        print('Y0 = {}'.format(first_horizontal_bottom_y + module_width / 2.))

        first_vertical_left_xs = [position[0] for position in block_vertical_left_plane_positions['block 1'][1]]
        first_vertical_left_x = sum(first_vertical_left_xs) / len(first_vertical_left_xs)
        print('first_vertical_left_x = {}'.format(first_vertical_left_x))
        print('X0 = {}'.format(first_vertical_left_x - module_width / 2.))

        gr_vertical = TGraph(len(vertical_planes), np.array(vertical_planes), np.array(vertical_left_xs))
        gr_horiztonal = TGraph(len(horizontal_planes), np.array(horizontal_planes), np.array(horizontal_bottom_ys))
        h1 = TH1D('h1', 'h1', 70, 2500, 2570)
        for module_width in module_widths:
            h1.Fill(module_width)

        # c1 = TCanvas('c1', 'c1', 600, 600)
        # set_margin()
        # gPad.SetGrid()
        # gStyle.SetOptStat('emr')
        # set_h1_style(h1)
        # h1.Draw()
        # h1.GetXaxis().SetTitle('Vertical Module Width (mm)')
        # h1.GetYaxis().SetTitle('Vertical Module Count')
        # c1.Update()
        # draw_statbox(h1, x1=0.65)
        # c1.Update()
        # c1.SaveAs('{}/plot_alignment_data_nova_detector_edge.module_width.pdf'.format(self.figure_dir))

        # c2 = TCanvas('c2', 'c2', 600, 600)
        # set_margin()
        # gPad.SetLeftMargin(0.2)
        # gPad.SetGrid()
        # set_graph_style(gr_vertical)
        # gr_vertical.Draw('AP')
        # gr_vertical.GetXaxis().SetTitle('Plane Number')
        # gr_vertical.GetYaxis().SetTitle('X (mm)')
        # gr_vertical.GetYaxis().SetTitleOffset(2.)
        # c2.Update()
        # c2.SaveAs('{}/plot_alignment_data_nova_detector_edge.vertical_left.pdf'.format(self.figure_dir))

        c3 = TCanvas('c3', 'c3', 600, 600)
        set_margin()
        gPad.SetLeftMargin(0.2)
        gPad.SetGrid()
        set_graph_style(gr_horiztonal)
        gr_horiztonal.Draw('AP')
        gr_horiztonal.GetXaxis().SetTitle('Plane Number')
        gr_horiztonal.GetYaxis().SetTitle('Y (mm)')
        gr_horiztonal.GetYaxis().SetTitleOffset(2.)
        c3.Update()
        # c3.SaveAs('{}/plot_alignment_data_nova_detector_edge.horizontal_bottom.left.pdf'.format(self.figure_dir))
        c3.SaveAs('{}/plot_alignment_data_nova_detector_edge.horizontal_bottom.pdf'.format(self.figure_dir))
        input('Press any key to continue.')

    def plot_alignment_data_nova_detector_vertical_center_block_1(self):
        name_positions = {}
        with open('alignment/Block 1 and Block 2 offset points Target frame.txt') as f_txt:
            rows = csv.reader(f_txt, delimiter=',')
            row_count = 0
            for i in range(5):
                next(rows)
            for row in rows:
                row_count += 1
                name = row[0].strip()
                x = float(row[-3])
                y = float(row[-2])
                z = float(row[-1])
                position = -x, z, y
                if name not in name_positions:
                    name_positions[name] = []
                name_positions[name].append(position)
            print('row_count = {}'.format(row_count))

        positions = name_positions['Block 1 Center Split points projected']

        xs = []
        ys = []
        zs = []
        for position in positions:
            xs.append(position[0])
            ys.append(position[1])
            zs.append(position[2])

        gr_xy = TGraph(len(xs), np.array(xs), np.array(ys))
        c1 = TCanvas('c1', 'c1', 600, 600)
        set_margin()
        gPad.SetLeftMargin(0.2)
        set_graph_style(gr_xy)
        gr_xy.Draw('AP')
        gr_xy.GetXaxis().SetTitle('X (mm)')
        gr_xy.GetYaxis().SetTitle('Y (mm)')
        gr_xy.GetYaxis().SetTitleOffset(2.)
        gr_xy.GetXaxis().SetNdivisions(405, 1)

        tf = TF1('tf1', 'pol1')
        gr_xy.Fit('tf1')

        latex = TLatex()
        latex.SetNDC()
        latex.SetTextFont(43)
        latex.SetTextSize(26)
        latex.DrawLatex(0.3, 0.85, 'f(x) = {:.3E} x {:.3E}'.format(tf.GetParameter(1), tf.GetParameter(0)))

        theta = atan(1. / tf.GetParameter(1)) * 180. / pi
        print('theta = {} degree'.format(theta))

        c1.Update()
        c1.SaveAs('{}/plot_alignment_data_nova_detector_vertical_center_block_1.gr_xy.pdf'.format(self.figure_dir))

        h_z = TH1D('h_z', 'h_z', 75, 14610, 14625)
        h_x = TH1D('h_x', 'h_x', 100, -1300, -1400)
        for position in positions:
            h_x.Fill(position[0])
            h_z.Fill(position[2])

        print('h_z.GetMean() = {}'.format(h_z.GetMean()))
        print('h_x.GetMean() = {}'.format(h_x.GetMean()))

        c2 = TCanvas('c2', 'c2', 600, 600)
        set_margin()
        set_h1_style(h_z)
        h_z.Draw()
        h_z.GetXaxis().SetTitle('Z (mm)')
        h_z.GetYaxis().SetTitle('Data Point Count')
        gStyle.SetOptStat('emr')
        c2.Update()
        draw_statbox(h_z, x1=0.6, y1=0.75)
        c2.Update()
        c2.SaveAs('{}/plot_alignment_data_nova_detector_vertical_center_block_1.h_z.pdf'.format(self.figure_dir))

        # c3 = TCanvas('c3', 'c3', 800, 600)
        # set_margin()
        # set_h1_style(h_x)
        # h_x.Draw()
        # h_x.GetXaxis().SetTitle('Z (mm)')
        # h_x.GetYaxis().SetTitle('Data Point Count')
        # gStyle.SetOptStat('emr')
        # c3.Update()
        # draw_statbox(h_x, x1=0.7)
        # c3.Update()
        # c3.SaveAs('{}/plot_alignment_data_nova_detector_vertical_center_block_1.h_x.pdf'.format(self.figure_dir))
        input('Press any key to continue.')

    def plot_alignment_data_nova_detector_front_surface(self):
        name_positions = {}
        with open('alignment/Block 1 and Block 2 offset points Target frame.txt') as f_txt:
            rows = csv.reader(f_txt, delimiter=',')
            row_count = 0
            for i in range(5):
                next(rows)
            for row in rows:
                row_count += 1
                name = row[0].strip()
                x = float(row[-3])
                y = float(row[-2])
                z = float(row[-1])
                position = -x, z, y
                if name not in name_positions:
                    name_positions[name] = []
                name_positions[name].append(position)
            print('row_count = {}'.format(row_count))

        positions = name_positions['Block 1 UPST Plane Offset Points']

        xs = []
        ys = []
        zs = []
        for position in positions:
            xs.append(position[0])
            ys.append(position[1])
            zs.append(position[2])

        gr_xz = TGraph(len(xs), np.array(xs), np.array(zs))
        c1 = TCanvas('c1', 'c1', 600, 600)
        set_margin()
        gPad.SetLeftMargin(0.2)
        set_graph_style(gr_xz)
        gr_xz.Draw('AP')
        gr_xz.GetXaxis().SetTitle('X (mm)')
        gr_xz.GetYaxis().SetTitle('Z (mm)')
        gr_xz.GetYaxis().SetTitleOffset(2.2)
        gr_xz.GetXaxis().SetNdivisions(505, 1)

        tf = TF1('tf1', 'pol1')
        gr_xz.Fit('tf1')

        latex = TLatex()
        latex.SetNDC()
        latex.SetTextFont(43)
        latex.SetTextSize(26)
        latex.DrawLatex(0.25, 0.83, 'f(x) = {:.3E} x + {:.3E}'.format(tf.GetParameter(1), tf.GetParameter(0)))

        theta = atan(tf.GetParameter(1)) * 180. / pi
        print('theta = {} degree'.format(theta))

        c1.Update()
        c1.SaveAs('{}/plot_alignment_data_nova_detector_front_surface.pdf'.format(self.figure_dir))
        input('Press any key to continue.')


# if __name__ == '__main__':
# gStyle.SetOptStat(0)
beamline = Beamline()
beamline.figure_dir = './figures'
# beamline.screen_shot = True
beamline.write()
# beamline.plot_position()

# 20180912_testbeam_radiation_collimator
# beamline = Beamline('tmp/beamline.py.radiation.collimator.in')
# beamline.write_radiation_collimator()

# 20180530_testbeam_radiation_dosage
# beamline = Beamline('tmp/beamline.py.radiation.dosage.in')
# beamline.write_radiation_dosage()

# 20181031_beamline_sim_update
# beamline = Beamline('tmp/beamline.py.geometry_check.in')
# beamline.write_geometry_check()

# 20190424_testbeam_alignment
# beamline = Beamline()
# beamline.figure_dir = '/Users/juntinghuang/beamer/20190424_testbeam_alignment/figures'
# beamline.screen_shot = True
# beamline.read_alignment_data_beamline()
# beamline.read_alignment_data_beamline_helium_pipe()
# beamline.write()
# beamline.print_flight_distance()
# beamline.read_alignment_data_beamline_collimator_us()
# beamline.read_alignment_data_beamline_mwpc()
# beamline.read_alignment_data_beamline_magnet()
# beamline.plot_alignment_data_nova_detector()
# beamline.plot_alignment_data_nova_detector_edge()
# beamline.plot_alignment_data_nova_detector_vertical_center_block_1()
# beamline.plot_alignment_data_nova_detector_front_surface()
# beamline = Beamline('tmp/beamline.py.geometry_check.in')
# beamline.read_alignment_data_beamline()
# beamline.read_alignment_data_beamline_helium_pipe()
# beamline.write()
# beamline.write_geometry_check()
# beamline.write_nova_plane()
# beamline.calculate()
# beamline.plot_position()
# beamline.plot_vertical_positions()
