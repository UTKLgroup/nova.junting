import csv
from rootalias import *
from math import pi, tan, sin, cos


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

    def __init__(self, g4bl_filename='beamline.py.in'):
        self.figure_dir = None
        self.us_theta = -16.    # degree
        self.ds_theta = 16.     # degree
        self.f_out = open(g4bl_filename, 'w')
        self.screen_shot = False
        # self.kill = 1
        self.kill = 0
        self.magnet_by = -0.9    # B field in tesla
        self.distance_target_to_ground = 83. * Beamline.INCH # estimated height of the beam pipe center with respect to the ground
        self.wire_chamber_support_length = 12.061 * Beamline.INCH
        self.grating_upstream_edge_z = 298.211 * Beamline.INCH

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
        self.shielding_block = Detector('shielding block')
        self.shielding_block_1 = Detector('shielding block 1')
        self.shielding_block_2 = Detector('shielding block 2')
        self.shielding_block_3 = Detector('shielding block 3')
        self.shielding_block_4 = Detector('shielding block 4')

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

        self.shielding_blocks = [
            self.shielding_block_1,
            self.shielding_block_2,
            self.shielding_block_3
        ]

        self.components = self.detectors + self.shielding_blocks

        self.read_position()
        self.correct_position()

        self.read_nova_dimension()
        self.read_magnet_dimension()
        # self.read_collimator_us_dimension()
        # self.read_collimator_ds_dimension()
        self.read_cherenkov_dimension()

    def __del__(self):
        self.f_out.close()

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

    def read_position(self):
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

    def read_nova_dimension(self):
        top_points = Beamline.get_csv('digitize/nova.csv')
        self.nova.length = np.average([Beamline.get_distance(top_points[0], top_points[1]), Beamline.get_distance(top_points[2], top_points[3])])
        self.nova.width = Beamline.get_distance(top_points[1], top_points[2])
        self.nova.height = self.nova.width

    def read_magnet_dimension(self):
        top_points = Beamline.get_csv('digitize/magnet.csv')
        self.magnet.length = np.average([Beamline.get_distance(top_points[0], top_points[1]), Beamline.get_distance(top_points[2], top_points[3])])
        self.magnet.width = Beamline.get_distance(top_points[1], top_points[2])
        self.magnet.aperture_width = Beamline.get_distance(top_points[4], top_points[5])

        side_points = Beamline.get_csv('digitize/magnet.side.csv')
        self.magnet.height = Beamline.get_distance(side_points[1], side_points[2])
        self.magnet.aperture_height = Beamline.get_distance(side_points[4], side_points[5])

    def read_collimator_ds_dimension(self):
        top_points = Beamline.get_csv('digitize/collimator_ds.csv')
        self.collimator_ds.length = (Beamline.get_distance(top_points[0], top_points[1]) + Beamline.get_distance(top_points[2], top_points[3])) / 2.
        self.collimator_ds.width = Beamline.get_distance(top_points[1], top_points[2])
        self.collimator_ds.aperture_width = Beamline.get_distance(top_points[4], top_points[5])

        side_points = Beamline.get_csv('digitize/collimator_ds.side.csv')
        self.collimator_ds.height = Beamline.get_distance(side_points[1], side_points[2])
        self.collimator_ds.aperture_height = Beamline.get_distance(side_points[4], side_points[5])

    def read_collimator_us_dimension(self):
        top_points = Beamline.get_csv('digitize/collimator_us.csv')
        self.collimator_ds.length = np.average([Beamline.get_distance(top_points[0], top_points[2]), Beamline.get_distance(top_points[3], top_points[5])])
        self.collimator_ds.width = np.average([Beamline.get_distance(top_points[2], top_points[3]), Beamline.get_distance(top_points[1], top_points[4])])

        side_points = Beamline.get_csv('digitize/collimator_us.side.csv')
        self.collimator_ds.height = np.average([Beamline.get_distance(side_points[0], side_points[5]), Beamline.get_distance(side_points[1], side_points[4])])
        self.collimator_ds.aperture_height = np.average([Beamline.get_distance(side_points[2], side_points[3]), Beamline.get_distance(side_points[6], side_points[7])])

    def read_cherenkov_dimension(self):
        top_points = Beamline.get_csv('digitize/ftbf_drawing_digitize.csv')
        self.cherenkov.length = self.get_distance(top_points[16], top_points[17])

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
            kOrange + 2,
            kBlue,
            kBlue,
            kBlue
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
            31,
            32,
            33,
            34
        ]

        c1 = TCanvas('c1', 'c1', 1200, 600)
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
        # gr.GetYaxis().SetRangeUser(-160, 50)
        gr.GetYaxis().SetRangeUser(-220, 120)
        gr.GetYaxis().SetNdivisions(505, 1)
        gr.GetXaxis().SetNdivisions(508, 1)

        # lg1 = TLegend(0.5, 0.33, 0.87, 0.86)
        # lg1.SetTextSize(22)
        lg1 = TLegend(0.34, 0.53, 0.87, 0.86)
        set_legend_style(lg1)
        lg1.SetNColumns(2)
        lg1.SetTextSize(18)
        lg1.SetMargin(0.15)
        lg1.SetBorderSize(1)


        # components = self.detectors + self.shielding_blocks
        print('len(self.components) = {}'.format(len(self.components)))
        print('len(colors) = {}'.format(len(colors)))
        print('len(styles) = {}'.format(len(styles)))

        markers = []
        for i, detector in enumerate(self.components):
            marker = TMarker(detector.z / 10., detector.x / 10., 24)
            markers.append(marker)
            markers[i].SetMarkerColor(colors[i])
            markers[i].SetMarkerStyle(styles[i])
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

    def write_virtual_disk(self):
        start_line_radius = 1750.
        start_line_length = 1.
        start_line_r = 1450.
        start_line_positions = [
            start_line_r * sin(self.us_theta * Beamline.RADIAN_PER_DEGREE),
            0.,
            start_line_r * cos(self.us_theta * Beamline.RADIAN_PER_DEGREE)
        ]
        self.f_out.write('virtualdetector start_line radius={} length={} material=Air color=0.9,0.9,0.7\n'.format(start_line_radius, start_line_length))
        self.f_out.write('place start_line rotation=y{} x={} y={} z={}\n'.format(self.us_theta, start_line_positions[0], start_line_positions[1], start_line_positions[2]))

    def write_wc(self):
        wire_chamber_detector_dimensions = [125., 25., 128.]
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
        self.wc_3.z = self.grating_upstream_edge_z - self.wire_chamber_support_length / 2.

        self.f_out.write('group wire_chamber\n')
        self.f_out.write('  virtualdetector wire_chamber_detector height={} length={} width={} color=0,1,0\n'.format(wire_chamber_detector_dimensions[0], wire_chamber_detector_dimensions[1], wire_chamber_detector_dimensions[2]))
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

    def write_magnet(self):
        self.magnet.height = 30. * Beamline.INCH

        magnet_field_dimensions = [3.5 * Beamline.INCH, 42 * Beamline.INCH, 17.75 * Beamline.INCH]
        magnet_iron_dimensions = [self.magnet.height, 42 * Beamline.INCH, 50 * Beamline.INCH]
        self.magnet.theta = self.us_theta + self.ds_theta / 2.

        self.f_out.write('genericbend M1 fieldHeight={} fieldLength={} fieldWidth={} kill={} ironColor=1,0,0 ironHeight={} ironLength={} ironWidth={}\n'.format(magnet_field_dimensions[0], magnet_field_dimensions[1], magnet_field_dimensions[2], self.kill, magnet_iron_dimensions[0], magnet_iron_dimensions[1], magnet_iron_dimensions[2]))
        self.f_out.write('place M1 By=$b_field x={} y={} z={} rotation=Y{}\n'.format(self.magnet.x, self.magnet.y, self.magnet.z, self.magnet.theta))

    def write_tof(self):
        tof_us_dimensions = [150., 20., 150.]
        self.tof_us.theta = self.us_theta
        tof_ds_dimensions = [150., 20., 150.]
        self.tof_ds.theta = self.us_theta + self.ds_theta

        # the downstream tof is about 2.375 inches to the front surface of the NOvA test beam detector
        self.tof_ds.z = self.nova.z - 2.375 * Beamline.INCH

        self.f_out.write('virtualdetector tof_us  height={} length={} width={} material=LUCITE color=0.05,0.05,0.93\n'.format(tof_us_dimensions[0], tof_us_dimensions[1], tof_us_dimensions[2]))
        self.f_out.write('place tof_us rename=tof_us x={} y={} z={} rotation=y{}\n'.format(self.tof_us.x, self.tof_us.y, self.tof_us.z, self.tof_us.theta))
        self.f_out.write('virtualdetector tof_ds height={} length={} width={} material=LUCITE color=0.05,0.05,0.93\n'.format(tof_ds_dimensions[0], tof_ds_dimensions[1], tof_ds_dimensions[2]))
        self.f_out.write('place tof_ds rename=tof_ds x={} y={} z={} rotation=y{}\n'.format(self.tof_ds.x, self.tof_ds.y, self.tof_ds.z, self.tof_ds.theta))

    def write_nova_plane(self):
        self.nova.theta = self.us_theta + self.ds_theta
        self.nova.length = 10.
        self.f_out.write('virtualdetector nova height={} length={} width={} color=0.39,0.39,0.39\n'.format(self.nova.height, self.nova.length, self.nova.width))
        self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y{}\n'.format(self.nova.x, self.nova.y, self.nova.z, self.nova.theta))
        print('self.nova.z = {}'.format(self.nova.z))

    def write_nova(self):
        self.nova.theta = self.us_theta + self.ds_theta
        self.nova.length = 3900.
        self.f_out.write('virtualdetector nova height={} length={} width={} material=POLYSTYRENE color=0.39,0.39,0.39\n'.format(self.nova.height, self.nova.length, self.nova.width))
        self.f_out.write('place nova rename=nova x={} y={} z={} rotation=y{}\n'.format(self.nova.x, self.nova.y, self.nova.z + self.nova.length / 2., self.nova.theta))

    def write_cherenkov(self):
        self.cherenkov.theta = self.us_theta + self.ds_theta
        self.cherenkov.length = 2925.
        cherenkov_inner_radius = 315. / 2.
        cherenkov_outer_radius = 324. / 2.
        cherenkov_pmt_pipe_outer_radius = 273. / 2.
        cherenkov_pmt_pipe_inner_radius = 264. / 2.
        cherenkov_pmt_pipe_length = 846. - 324. / 2.
        support_dimensions = [4.5 * Beamline.FOOT, 6. * Beamline.FOOT, 3. * Beamline.FOOT]

        # the distance between wire chamber 4 and the upstream end of the Cerenkov counter is about 48 inches
        self.cherenkov.z = self.wc_4.z + 48. * Beamline.INCH + self.cherenkov.length / 2.
        cherenkov_end_to_support_edge_distance = 3.625 * Beamline.INCH
        cherenkov_to_support_distance = 6.125 * Beamline.INCH

        self.f_out.write('virtualdetector cherenkov radius={} length={} color=1,1,1 material=CARBON_DIOXIDE\n'.format(cherenkov_inner_radius, self.cherenkov.length))
        self.f_out.write('tubs cherenkov_pipe innerRadius={} outerRadius={} length={} color=0.74,0.34,0.09 material=STAINLESS-STEEL\n'.format(cherenkov_inner_radius, cherenkov_outer_radius, self.cherenkov.length))
        self.f_out.write('tubs cherenkov_pipe_pmt innerRadius={} outerRadius={} length={} color=0.74,0.34,0.09 material=STAINLESS-STEEL\n'.format(cherenkov_pmt_pipe_inner_radius, cherenkov_pmt_pipe_outer_radius, cherenkov_pmt_pipe_length))
        self.f_out.write('box cherenkov_support height={} length={} width={} material=CONCRETE color=0,1,1 kill={}\n'.format(support_dimensions[0], support_dimensions[1], support_dimensions[2], self.kill))

        self.f_out.write('place cherenkov rename=cherenkov x={} y={} z={} rotation=y{}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z, self.cherenkov.theta))
        self.f_out.write('place cherenkov_pipe rename=cherenkov_pipe x={} y={} z={} rotation=y{} kill={}\n'.format(self.cherenkov.x, self.cherenkov.y, self.cherenkov.z, self.cherenkov.theta, self.kill))
        self.f_out.write('place cherenkov_pipe_pmt rename=cherenkov_pipe_pmt x={} y={} z={} rotation=y{},x90 kill={}\n'.format(self.cherenkov.x, self.cherenkov.y - cherenkov_outer_radius - cherenkov_pmt_pipe_length / 2., self.cherenkov.z + self.cherenkov.length / 2. - cherenkov_pmt_pipe_outer_radius - 1.75 * Beamline.INCH, self.cherenkov.theta, self.kill))
        self.f_out.write('place cherenkov_support x={} y={} z={} rotation=y{}\n'.format(self.cherenkov.x, self.cherenkov.y - cherenkov_outer_radius - cherenkov_to_support_distance - support_dimensions[0] / 2., self.cherenkov.z - self.cherenkov.length / 2. + cherenkov_end_to_support_edge_distance + support_dimensions[1] / 2., self.cherenkov.theta))

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

        # shielding_block_separation = 20. * Beamline.INCH
        # self.shielding_block_1.set_zx([self.magnet.z + 1200., self.magnet.x + self.shielding_block.width / 2. + shielding_block_separation / 2.])
        # self.shielding_block_2.set_zx([self.magnet.z + 1200., self.magnet.x - self.shielding_block.width / 2. - shielding_block_separation / 2.])
        # self.shielding_block_1.set_zx([self.collimator_ds.z, self.collimator_ds.x + self.shielding_block.width / 2. + self.collimator_ds.width / 2.])
        self.shielding_block_1.set_zx([self.wc_3.z - self.wire_chamber_support_length / 2. - self.shielding_block.length / 2., self.wc_3.x + 5.789 * Beamline.INCH + self.shielding_block.width / 2.])
        # self.shielding_block_2.set_zx([self.collimator_ds.z, self.collimator_ds.x - self.shielding_block.width / 2. - self.collimator_ds.width / 2.])
        # self.shielding_block_2.set_zx([self.wc_3.z, self.wc_3.x - 17.676 * Beamline.INCH - self.shielding_block.width / 2.])
        self.shielding_block_2.set_zx([self.wc_3.z - self.wire_chamber_support_length / 2. - self.shielding_block.length / 2., self.wc_3.x - 5.789 * Beamline.INCH - self.shielding_block.width / 2.])
        # self.shielding_block_3.set_zx([self.shielding_block_1.z, self.shielding_block_1.x + self.shielding_block.width])
        self.shielding_block_3.set_zx([2700. - (self.collimator_us.length - (42.76 + 8.36) * Beamline.INCH) + self.shielding_block.length / 2., (7.47 * Beamline.INCH + 460.) - (420. + 12. * Beamline.INCH)])
        self.shielding_block_4.set_zx([self.collimator_ds.z - self.collimator_ds.length / 2. + 15.647 * Beamline.INCH + self.shielding_block.length / 2., self.collimator_ds.x - self.shielding_block.width / 2. - self.collimator_ds.width / 2.])

        # the distance from ground to the top of the shielding block is 2.7m
        y_shift = (2700. - 16. * Beamline.INCH) - self.distance_target_to_ground
        self.shielding_block_1.y += y_shift
        self.shielding_block_2.y += y_shift
        self.shielding_block_3.y += y_shift
        self.shielding_block_4.y += y_shift

        self.f_out.write('place shielding_block rename=shielding_block_1 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_1.x, self.shielding_block_1.y, self.shielding_block_1.z, self.shielding_block_1.theta))
        self.f_out.write('place shielding_block rename=shielding_block_2 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_2.x, self.shielding_block_2.y, self.shielding_block_2.z, self.shielding_block_2.theta))
        self.f_out.write('place shielding_block rename=shielding_block_3 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_3.x, self.shielding_block_3.y, self.shielding_block_3.z, self.shielding_block_3.theta))
        # self.f_out.write('place shielding_block rename=shielding_block_4 x={} y={} z={} rotation=y{}\n'.format(self.shielding_block_4.x, self.shielding_block_4.y, self.shielding_block_4.z, self.shielding_block_4.theta))

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

    def correct_position(self):
        us_detectors = [
            self.tof_us,
            self.wc_1,
            self.wc_2
        ]
        for us_detector in us_detectors:
            distance = us_detector.get_r()
            us_detector.x = sin(self.us_theta * Beamline.RADIAN_PER_DEGREE) * distance
            us_detector.z = cos(self.us_theta * Beamline.RADIAN_PER_DEGREE) * distance

        ds_detectors = [
            self.wc_3,
            self.collimator_ds,
            self.wc_4,
            self.cherenkov,
            self.tof_ds,
            self.nova
        ]
        ds_detector_average_x = np.average([ds_detector.x for ds_detector in ds_detectors])
        for ds_detector in ds_detectors:
            ds_detector.x = ds_detector_average_x

        self.magnet.x = ds_detector_average_x
        self.magnet.z = ds_detector_average_x / tan(self.us_theta * Beamline.RADIAN_PER_DEGREE)

    def write(self):
        self.f_out.write('physics QGSP_BIC\n')
        self.f_out.write('param worldMaterial=Air\n')
        self.f_out.write('param histoFile=beam.root\n')

        self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector 0 1 0"\n')
        # self.f_out.write('g4ui when=4 "/vis/viewer/set/viewpointVector -1 1 1"\n') # for radiation study
        self.f_out.write('g4ui when=4 "/vis/viewer/zoom 1.5"\n')
        self.f_out.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        if self.screen_shot:
            self.f_out.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')

        self.f_out.write('beam gaussian particle=$particle firstEvent=$first lastEvent=$last sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=$momentum\n')
        self.f_out.write('trackcuts kineticEnergyCut=20 keep=gamma,pi0,pi+,pi-,kaon+,kaon-,mu+,mu-,e+,e-,proton,anti_proton\n')
        # self.f_out.write('trackcuts kineticEnergyCut=5 keep=pi+,pi-,kaon+,kaon-,mu+,mu-,e+,e-,proton,anti_proton\n')
        # self.f_out.write('trackcuts kineticEnergyCut=5 kineticEnergyMax=3000 keep=pi+,pi-,kaon+,kaon-,mu+,mu-,e+,e-,proton,anti_proton\n')
        # self.f_out.write('trackcuts keep=pi+,pi-,pi0,kaon+,kaon-,mu+,mu-,e+,e-,gamma,proton,anti_proton\n')
        # self.f_out.write('trackcuts keep=pi+,pi-,pi0,kaon+,kaon-,mu+,mu-,proton,anti_proton,neutron,anti_neutron\n') # for radiation study

        self.write_target()
        self.write_collimator_us()
        if not self.screen_shot:
            self.write_virtual_disk()
        self.write_wc()
        self.write_magnet()
        self.write_collimator_ds()
        self.write_tof()
        self.write_cherenkov()
        self.write_nova_plane()
        self.write_shielding_block()
        # self.write_nova()       # for radiation study
        # self.write_housing()    # for radiation study

    def write_radiation(self):
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
        self.shielding_block_1.x = 0.
        self.shielding_block_1.y = 0.
        self.shielding_block_1.z = 0.
        self.shielding_block_1.theta = 0.
        self.write_shielding_block()


    def read_alignment_data_beamline(self):
        for component in self.components:
            if component.name not in ['nova detector', 'upstream collimator']:
                component.set_xyz((0., 0., 0.))

        with open('data/alignment/NTB summary_up_ct_dn.txt') as f_txt:
            for row in csv.reader(f_txt, delimiter=','):
                detector_name = row[1].strip()
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                position = -x, z, y

                if detector_name == 'NTB-CERENKOV_DN':
                    self.cherenkov.set_xyz(position)
                    self.cherenkov.z = 546.554
                    self.cherenkov.z += 15.75 - 134.86 / 2.

                if '_CT' in detector_name:
                    # print('detector = {}'.format(detector))
                    # print('x = {}, y = {}, z = {}'.format(x, y, z))
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

    def read_alignment_data_nova_detector(self):
        self.nova.set_xyz((0., 0., 0.))
        with open('data/alignment/NTB summary_up_ct_dn.txt') as f_txt:
            for row in csv.reader(f_txt, delimiter=','):
                # print('row = {}'.format(row))
                detector_name = row[1].strip()
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                if '_CT' in detector_name:
                    # print('detector = {}'.format(detector))
                    # print('x = {}, y = {}, z = {}'.format(x, y, z))
                    position = -x, z, y



# if __name__ == '__main__':
beamline = Beamline()
# beamline.figure_dir = '/Users/juntinghuang/beamer/20180413_testbeam_120gev/figures'
beamline.figure_dir = '/Users/juntinghuang/beamer/20190424_testbeam_alignment/figures'
# beamline.plot_position()
# beamline.screen_shot = True
# beamline.read_cherenkov_dimension()
beamline.write()
print('before: beamline.nova.z = {}'.format(beamline.nova.z))
beamline.read_alignment_data_beamline()
print('after: beamline.nova.z = {}'.format(beamline.nova.z))
# beamline.read_alignment_data_nova_detector()
beamline.plot_position()

# beamline = Beamline('beamline.py.radiation.collimator.in')
# beamline.write_radiation()

# beamline = Beamline('tmp/beamline.py.geometry_check.in')
# beamline.write_geometry_check()
