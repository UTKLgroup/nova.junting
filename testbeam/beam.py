from rootalias import *
from math import cos, sin, tan, pi


collimator_upstream_theta = -13. # degree
magnet_theta_relative = 10.      # degree

inch = 25.4
kill = 1
radian_per_degree = pi / 180.

target_positions = [0., 0., 0.]
target_slab_dimensions = [31.75, 209.55, 6.35] # [height, length, width]
target_slab_count = 5.
target_delta_x = target_slab_dimensions[0] / target_slab_count
target_delta_z = 22.145

collimator_upstream_base_dimensions = [5.19 * inch, 58. * inch, 32. * inch]
collimator_upstream_bottom_dimensions = [5.19 / 2. * inch, 42.76 * inch, 32. * inch]
collimator_upstream_middle_dimensions = [2. * inch, 42.76 * inch, 11.6 * inch]
collimator_upstream_top_dimensions = [5.19 * inch, 42.76 * inch, 32. * inch]
collimator_upstream_base_positions = [0., -6.19 * inch, 0.]
collimator_upstream_bottom_positions = [0., -(1. + 5.19 / 4.) * inch, 7.62 * inch]
collimator_upstream_middle_1_positions = [296. / 2. + 67.29, 0., 7.62 * inch]
collimator_upstream_middle_2_positions = [-296. / 2. - 67.29, 0., 7.62 * inch]
collimator_upstream_top_positions = [0., (1. + 5.19 / 2.) * inch, 7.62 * inch]
collimator_upstream_positions = [0., 0., (29. / 2. + 7.62) * inch]
collimator_upstream_base_theta = 3.               # degree, positive here means a counter-clockwise rotation in the top view
collimator_upstream_theta_offset = 1.97
collimator_upstream_middle_1_theta = collimator_upstream_theta + collimator_upstream_theta_offset
collimator_upstream_middle_2_theta = collimator_upstream_theta - collimator_upstream_theta_offset
collimator_upstream_parts = [
    collimator_upstream_base_positions,
    collimator_upstream_bottom_positions,
    collimator_upstream_middle_1_positions,
    collimator_upstream_middle_2_positions,
    collimator_upstream_top_positions
]

magnet_field_dimensions = [3.5 * inch, 42 * inch, 17.75 * inch]
magnet_iron_dimensions = [28. * inch, 42. * inch, 42. * inch]
magnet_by = 1.8                 # B field, tesla
magnet_positions = [-945.7, 0., 4228.9]
magnet_theta = collimator_upstream_theta + magnet_theta_relative / 2.

tof_upstream_dimensions = [150., 50.8, 150.]
tof_upstream_positions = [-346.54341, 0., 1423.]
tof_upstream_theta = collimator_upstream_theta
tof_downstream_dimensions = [130., 50.8, 130.]
tof_downstream_positions = [-1186.1546, 0., 8005.9022]
tof_downstream_theta = collimator_upstream_theta + magnet_theta_relative

wire_chamber_detector_dimensions = [125., 25., 128.]
wire_chamber_frame_vertical_dimensions = [254., 25., 63.]
wire_chamber_frame_horizontal_dimensions = [63., 25., 128.]
wire_chamber_detector_positions = [0., 0., 12.5]
wire_chamber_frame_vertical_left_positions = [-95.5, 0., 12.5]
wire_chamber_frame_vertical_right_positions = [95.5, 0., 12.5]
wire_chamber_frame_horizontal_top_positions = [0., 95.5, 12.5]
wire_chamber_frame_horizontal_bottom_positions = [0., -95.5, 12.5]
wire_chamber_1_positions = [-403.0472, 0.0508, 1730.3369]
wire_chamber_2_positions = [-738.0351, 0.0762, 3181.9215]
wire_chamber_3_positions = [-1064.133, -2.921, 5167.5665]
wire_chamber_4_positions = [-1195.0827, -20.4724, 7606.4872]
wire_chamber_1_theta = collimator_upstream_theta
wire_chamber_2_theta = collimator_upstream_theta
wire_chamber_3_theta = collimator_upstream_theta + magnet_theta_relative
wire_chamber_4_theta = collimator_upstream_theta + magnet_theta_relative

collimator_downstream_bottom_dimensions = [8.5 * inch, 36. * inch, 30. * inch]
collimator_downstream_middle_dimensions = [6. * inch, 36. * inch, 11. * inch]
collimator_downstream_middle_1_positions=[9. * inch, 0., 18. * inch]
collimator_downstream_middle_2_positions = [-9. * inch, 0., 18. * inch ]
collimator_downstream_bottom_positions = [0., (4.25 + 3.) * inch, 18. * inch]
collimator_downstream_top_positions = [0., -(4.25 + 3.) * inch, 18. * inch]
collimator_downstream_positions = [-1077., 0., 6188. - 9. * inch]
collimator_downstream_theta = collimator_upstream_theta + magnet_theta_relative

start_line_radius = 1750.
start_line_length = 1.
start_line_theta = collimator_upstream_theta
start_line_positions = [-335. + 5.7, 0., 1400. - 26.4]


def translate(positions, deltas):
    for i in range(len(positions)):
        positions[i] += deltas[i]


def rotate_y(positions, x0, z0, theta):
    # theta is positive for counter-clockwise rotation around (x0, z0) in topview (looking along negative y direction)
    x = positions[0]
    z = positions[2]

    x -= x0
    z -= z0
    z_rotate = cos(theta) * z - sin(theta) * x
    x_rotate = sin(theta) * z + cos(theta) * x
    x_rotate += x0
    z_rotate += z0

    positions[0] = x_rotate
    positions[2] = z_rotate


def rotate_y_absolute(positions, theta, distance, axis):
    # theta in degree
    positions[0] = sin(theta * radian_per_degree) * distance + axis[0]
    positions[2] = cos(theta * radian_per_degree) * distance + axis[2]


def get_distance(v, v0):
    return ((v[0] - v0[0])**2 + (v[1] - v0[1])**2 + (v[2] - v0[2])**2)**0.5


def move_collimator_upstream():
    # move z, rotate
    for collimator_upstream_part in collimator_upstream_parts:
        translate(collimator_upstream_part, collimator_upstream_positions)
        rotate_y(collimator_upstream_part, collimator_upstream_positions[0], collimator_upstream_positions[2] + 29. / 2. * inch, collimator_upstream_base_theta * pi / 180.)

    # move x
    collimator_upstream_middle_z = (collimator_upstream_middle_1_positions[2] + collimator_upstream_middle_2_positions[2]) / 2.
    offset = tan(collimator_upstream_theta * pi / 180.) * collimator_upstream_middle_z
    for collimator_upstream_part in collimator_upstream_parts:
        translate(collimator_upstream_part, [offset, 0., 0.])


def rotate_updownstream():
    start_line_distance = get_distance(start_line_positions, target_positions) + 10.
    tof_upstream_distance = get_distance(tof_upstream_positions, target_positions)
    wire_chamber_1_distance = get_distance(wire_chamber_1_positions, target_positions)
    wire_chamber_2_distance = get_distance(wire_chamber_2_positions, target_positions)
    magnet_distance = get_distance(magnet_positions, target_positions)

    wire_chamber_3_distance = get_distance(wire_chamber_3_positions, magnet_positions)
    collimator_downstream_distance = get_distance(collimator_downstream_positions, magnet_positions)
    wire_chamber_4_distance = get_distance(wire_chamber_4_positions, magnet_positions)
    tof_downstream_distance = get_distance(tof_downstream_positions, magnet_positions)

    rotate_y_absolute(start_line_positions, collimator_upstream_theta, start_line_distance, target_positions)
    rotate_y_absolute(tof_upstream_positions, tof_upstream_theta, tof_upstream_distance, target_positions)
    rotate_y_absolute(wire_chamber_1_positions, wire_chamber_1_theta, wire_chamber_1_distance, target_positions)
    rotate_y_absolute(wire_chamber_2_positions, wire_chamber_2_theta, wire_chamber_2_distance, target_positions)
    rotate_y_absolute(magnet_positions, collimator_upstream_theta, magnet_distance, target_positions)

    rotate_y_absolute(wire_chamber_3_positions, wire_chamber_3_theta, wire_chamber_3_distance, magnet_positions)
    rotate_y_absolute(collimator_downstream_positions, collimator_downstream_theta, collimator_downstream_distance, magnet_positions)
    rotate_y_absolute(wire_chamber_4_positions, wire_chamber_4_theta, wire_chamber_4_distance, magnet_positions)
    rotate_y_absolute(tof_downstream_positions, tof_downstream_theta, tof_downstream_distance, magnet_positions)


def write():
    # with open('beam.py.in', 'w') as f_beam:
    with open('tmp/beam.py.theta_t_{}.theta_m_{}.in'.format(collimator_upstream_theta, magnet_theta_relative), 'w') as f_beam:
        f_beam.write('physics QGSP_BIC\n')
        f_beam.write('param worldMaterial=Air\n')
        f_beam.write('param histoFile=beam.root\n')

        f_beam.write('g4ui when=4 "/vis/viewer/set/viewpointVector 0 1 0"\n')
        f_beam.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        # f_beam.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')

        f_beam.write('beam gaussian particle=pi+ firstEvent=$first lastEvent=$last sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=$momentum\n')
        f_beam.write('trackcuts keep=pi+,pi-,pi0,kaon+,kaon-,mu+,mu-,e+,e-,gamma,proton,anti_proton\n')

        f_beam.write('box slab height={} length={} width={} material=Cu color=1,0.01,0.01\n'.format(target_slab_dimensions[0], target_slab_dimensions[1], target_slab_dimensions[2]))
        for i in range(-2, 3):
            f_beam.write('place slab rename=target_slab_{} x={} y={} z={}\n'.format(i, target_positions[0] + i * target_delta_x, target_positions[1], -i * target_delta_z + target_positions[2]))

        f_beam.write('box collimator_upstream_base height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_base_dimensions[0], collimator_upstream_base_dimensions[1], collimator_upstream_base_dimensions[2], kill))
        f_beam.write('box collimator_upstream_bottom height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_bottom_dimensions[0], collimator_upstream_bottom_dimensions[1], collimator_upstream_bottom_dimensions[2], kill))
        f_beam.write('box collimator_upstream_middle height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_middle_dimensions[0], collimator_upstream_middle_dimensions[1], collimator_upstream_middle_dimensions[2], kill))
        f_beam.write('box collimator_upstream_top height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_top_dimensions[0], collimator_upstream_top_dimensions[1], collimator_upstream_top_dimensions[2], kill))

        f_beam.write('place collimator_upstream_base rename=collimator_upstream_base x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_base_positions[0], collimator_upstream_base_positions[1], collimator_upstream_base_positions[2], collimator_upstream_base_theta))
        f_beam.write('place collimator_upstream_bottom rename=collimator_upstream_bottom x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_bottom_positions[0], collimator_upstream_bottom_positions[1], collimator_upstream_bottom_positions[2], collimator_upstream_base_theta))
        f_beam.write('place collimator_upstream_middle rename=collimator_upstream_middle_1 x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_middle_1_positions[0], collimator_upstream_middle_1_positions[1], collimator_upstream_middle_1_positions[2], collimator_upstream_middle_1_theta))
        f_beam.write('place collimator_upstream_middle rename=collimator_upstream_middle_2 x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_middle_2_positions[0], collimator_upstream_middle_2_positions[1], collimator_upstream_middle_2_positions[2], collimator_upstream_middle_2_theta))
        f_beam.write('place collimator_upstream_top rename=collimator_upstream_top x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_top_positions[0], collimator_upstream_top_positions[1], collimator_upstream_top_positions[2], collimator_upstream_base_theta))

        f_beam.write('virtualdetector start_line radius={} length={} material=Air color=0.9,0.9,0.7\n'.format(start_line_radius, start_line_length))
        f_beam.write('place start_line rotation=y{} x={} y={} z={}\n'.format(start_line_theta, start_line_positions[0], start_line_positions[1], start_line_positions[2]))

        f_beam.write('virtualdetector tof_upstream  height={} length={} width={} material=LUCITE color=0.05,0.05,0.93\n'.format(tof_upstream_dimensions[0], tof_upstream_dimensions[1], tof_upstream_dimensions[2]))
        f_beam.write('place tof_upstream rename=tof_upstream x={} y={} z={} rotation=z45,y{}\n'.format(tof_upstream_positions[0], tof_upstream_positions[1], tof_upstream_positions[2], tof_upstream_theta))

        f_beam.write('group wire_chamber\n')
        f_beam.write('  virtualdetector wire_chamber_detector height={} length={} width={} color=0,1,0\n'.format(wire_chamber_detector_dimensions[0], wire_chamber_detector_dimensions[1], wire_chamber_detector_dimensions[2]))
        f_beam.write('  box wire_chamber_frame_vertical height={} length={} width={} color=1,0,1 kill={} material=Al\n'.format(wire_chamber_frame_vertical_dimensions[0], wire_chamber_frame_vertical_dimensions[1], wire_chamber_frame_vertical_dimensions[2], kill))
        f_beam.write('  box wire_chamber_frame_horizontal height={} length={} width={} color=1,0,1 kill={} material=Al\n'.format(wire_chamber_frame_horizontal_dimensions[0], wire_chamber_frame_horizontal_dimensions[1], wire_chamber_frame_horizontal_dimensions[2], kill))
        f_beam.write('  place wire_chamber_frame_vertical rename=+_frame_left x={} y={} z={}\n'.format(wire_chamber_frame_vertical_left_positions[0], wire_chamber_frame_vertical_left_positions[1], wire_chamber_frame_vertical_left_positions[2]))
        f_beam.write('  place wire_chamber_frame_vertical rename=+_frame_right x={} y={} z={}\n'.format(wire_chamber_frame_vertical_right_positions[0], wire_chamber_frame_vertical_right_positions[1], wire_chamber_frame_vertical_right_positions[2]))
        f_beam.write('  place wire_chamber_frame_horizontal rename=+_frame_top x={} y={} z={}\n'.format(wire_chamber_frame_horizontal_top_positions[0], wire_chamber_frame_horizontal_top_positions[1], wire_chamber_frame_horizontal_top_positions[2]))
        f_beam.write('  place wire_chamber_frame_horizontal rename=+_frame_bottom x={} y={} z={}\n'.format(wire_chamber_frame_horizontal_bottom_positions[0], wire_chamber_frame_horizontal_bottom_positions[1], wire_chamber_frame_horizontal_bottom_positions[2]))
        f_beam.write('  place wire_chamber_detector rename=+_detector x={} y={} z={}\n'.format(wire_chamber_detector_positions[0], wire_chamber_detector_positions[1], wire_chamber_detector_positions[2]))
        f_beam.write('endgroup\n')
        f_beam.write('place wire_chamber rename=wire_chamber_1 x={} y={} z={} rotation=y{}\n'.format(wire_chamber_1_positions[0], wire_chamber_1_positions[1], wire_chamber_1_positions[2], wire_chamber_1_theta))
        f_beam.write('place wire_chamber rename=wire_chamber_2 x={} y={} z={} rotation=y{}\n'.format(wire_chamber_2_positions[0], wire_chamber_2_positions[1], wire_chamber_2_positions[2], wire_chamber_2_theta))

        f_beam.write('genericbend M1 fieldHeight={} fieldLength={} fieldWidth={} kill={} ironColor=1,0,0 ironHeight={} ironLength={} ironWidth={}\n'.format(magnet_field_dimensions[0], magnet_field_dimensions[1], magnet_field_dimensions[2], kill, magnet_iron_dimensions[0], magnet_iron_dimensions[1], magnet_iron_dimensions[2]))
        f_beam.write('place M1 By={} x={} y={} z={} rotation=Y{}\n'.format(magnet_by, magnet_positions[0], magnet_positions[1], magnet_positions[2], magnet_theta))

        f_beam.write('place wire_chamber rename=wire_chamber_3 x={} y={} z={} rotation=y{}\n'.format(wire_chamber_3_positions[0], wire_chamber_3_positions[1], wire_chamber_3_positions[2], wire_chamber_3_theta))

        f_beam.write('group collimator_downstream\n')
        f_beam.write('  box collimator_downstream_bottom height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_downstream_bottom_dimensions[0], collimator_downstream_bottom_dimensions[1], collimator_downstream_bottom_dimensions[2], kill))
        f_beam.write('  box collimator_downstream_middle height={} length={} width={} material=Fe color=0,.8,1 kill={}\n'.format(collimator_downstream_middle_dimensions[0], collimator_downstream_middle_dimensions[1], collimator_downstream_middle_dimensions[2], kill))
        f_beam.write('  place collimator_downstream_middle rename=+_middle_1 x={} y={} z={}\n'.format(collimator_downstream_middle_1_positions[0], collimator_downstream_middle_1_positions[1], collimator_downstream_middle_1_positions[2]))
        f_beam.write('  place collimator_downstream_middle rename=+_middle_2 x={} y={} z={}\n'.format(collimator_downstream_middle_2_positions[0], collimator_downstream_middle_2_positions[1], collimator_downstream_middle_2_positions[2]))
        f_beam.write('  place collimator_downstream_bottom rename=+_bottom x={} y={} z={}\n'.format(collimator_downstream_bottom_positions[0], collimator_downstream_bottom_positions[1], collimator_downstream_bottom_positions[2]))
        f_beam.write('  place collimator_downstream_bottom rename=+_top x={} y={} z={}\n'.format(collimator_downstream_top_positions[0], collimator_downstream_top_positions[1], collimator_downstream_top_positions[2]))
        f_beam.write('endgroup\n')
        f_beam.write('place collimator_downstream x={} y={} z={} rotation=y{}\n'.format(collimator_downstream_positions[0], collimator_downstream_positions[1], collimator_downstream_positions[2], collimator_downstream_theta))

        f_beam.write('place wire_chamber rename=wire_chamber_4 x={} y={} z={} rotation=y{}\n'.format(wire_chamber_4_positions[0], wire_chamber_4_positions[1], wire_chamber_4_positions[2], wire_chamber_4_theta))

        f_beam.write('virtualdetector tof_downstream height={} length={} width={} material=LUCITE color=0.05,0.05,0.93\n'.format(tof_downstream_dimensions[0], tof_downstream_dimensions[1], tof_downstream_dimensions[2]))
        f_beam.write('place tof_downstream rename=tof_downstream x={} y={} z={} rotation=z90,y{}\n'.format(tof_downstream_positions[0], tof_downstream_positions[1], tof_downstream_positions[2], tof_downstream_theta))


FIGURE_DIR = '/Users/juntinghuang/beamer/20180211_testbeam_high_stat/figures'
def plot_position():
    positionss = [
        target_positions,
        collimator_upstream_base_positions,
        tof_upstream_positions,
        wire_chamber_1_positions,
        wire_chamber_2_positions,
        magnet_positions,
        wire_chamber_3_positions,
        collimator_downstream_positions,
        wire_chamber_4_positions,
        tof_downstream_positions
    ]

    names = [
        'target',
        'upstream collimator',
        'upstream TOF',
        'wire_chamber 1',
        'wire_chamber 2',
        'M1 magnet',
        'wire_chamber 3',
        'downstream collimator',
        'wire_chamber 4',
        'downstream TOF'
    ]

    colors = [
        kBlack,
        kRed + 2,
        kMagenta + 2,
        kViolet + 2,
        kBlue + 2,
        kAzure + 2,
        kCyan + 2,
        kTeal + 2,
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
        29
    ]

    c1 = TCanvas('c1', 'c1', 1200, 600)
    set_margin()
    gPad.SetTickx()
    gPad.SetTicky()

    gr = TGraph(2, np.array([-50., 1000.]), np.array([-1000., 1000.]))
    set_graph_style(gr)
    gr.SetMarkerSize(0)
    gr.SetLineWidth(0)
    gr.GetXaxis().SetTitle('Z (cm)')
    gr.GetYaxis().SetTitle('X (cm)')
    gr.Draw('AP')
    gr.GetXaxis().SetRangeUser(-50, 1200)
    gr.GetYaxis().SetRangeUser(-150, 50)
    gr.GetYaxis().SetNdivisions(505, 1)
    gr.GetXaxis().SetNdivisions(508, 1)

    lg1 = TLegend(0.51, 0.38, 0.87, 0.86)
    set_legend_style(lg1)
    lg1.SetTextSize(22)
    lg1.SetMargin(0.15)
    lg1.SetBorderSize(1)

    markers = []
    # latex = TLatex()
    # latex.SetTextFont(43)
    # latex.SetTextSize(15)
    for i, positions in enumerate(positionss):
        print('positions = {}'.format(positions))
        coordinate_x = positions[2] / 10. # cm
        coordinate_y = positions[0] / 10. # cm

        marker = TMarker(coordinate_x, coordinate_y, 24)
        markers.append(marker)
        markers[i].SetMarkerColor(colors[i])
        markers[i].SetMarkerStyle(styles[i])
        markers[i].SetMarkerSize(2.)
        markers[i].Draw()

        name = '{} ({:.1f}, {:.1f})'.format(names[i], coordinate_x, coordinate_y)
        lg1.AddEntry(markers[i], name, 'p')

    lg1.Draw()
    c1.Update()
    c1.SaveAs('{}/plot_position.theta_t_{}.theta_m_{}.pdf'.format(FIGURE_DIR, collimator_upstream_theta, magnet_theta_relative))
    input('Press any key to continue.')

move_collimator_upstream()
rotate_updownstream()
# write()
plot_position()
