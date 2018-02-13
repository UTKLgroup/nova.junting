from math import cos, sin, pi

inch = 25.4
kill = 1

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
collimator_upstream_positions = [-8.315 * inch + 40., 0., (29. / 2. + 7.62) * inch]
collimator_upstream_base_theta = 3               # degree, positive here means a counter-clockwise rotation in the top view
collimator_upstream_theta = -13
collimator_upstream_theta_offset = 1.97
collimator_upstream_middle_1_theta = collimator_upstream_theta + collimator_upstream_theta_offset
collimator_upstream_middle_2_theta = collimator_upstream_theta - collimator_upstream_theta_offset

tof_upstream_dimensions = [150., 50.8, 150.]
tof_upstream_positions = [-346.54341, 0., 1423.]
tof_upstream_theta = -13
tof_downstream_dimensions = [130., 50.8, 130.]
tof_downstream_positions = [-1186.1546, 0., 8005.9022]
tof_downstream_theta = -3

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
wire_chamber_1_theta = -13
wire_chamber_2_theta = -13
wire_chamber_3_theta = -3
wire_chamber_4_theta = -3

magnet_field_dimensions = [3.5 * inch, 42 * inch, 5.6 * inch]
magnet_iron_dimensions = [28. * inch, 42. * inch, 42. * inch]
magnet_by = 1.8
magnet_positions = [-945.7, 0., 4228.9]
magnet_theta = -13 + 5

collimator_downstream_bottom_dimensions = [8.5 * inch, 36. * inch, 30. * inch]
collimator_downstream_middle_dimensions = [6. * inch, 36. * inch, 11. * inch]
collimator_downstream_middle_1_positions=[9. * inch, 0., 18. * inch]
collimator_downstream_middle_2_positions = [-9. * inch, 0., 18. * inch ]
collimator_downstream_bottom_positions = [0., (4.25 + 3.) * inch, 18. * inch]
collimator_downstream_top_positions = [0., -(4.25 + 3.) * inch, 18. * inch]
collimator_downstream_positions = [-1077., 0., 6188. - 9. * inch]
collimator_downstream_theta = -3


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


def write():
    collimator_upstream_parts = [
        collimator_upstream_base_positions,
        collimator_upstream_bottom_positions,
        collimator_upstream_middle_1_positions,
        collimator_upstream_middle_2_positions,
        collimator_upstream_top_positions
       ]
    
    for collimator_upstream_part in collimator_upstream_parts:
        translate(collimator_upstream_part, collimator_upstream_positions)
        rotate_y(collimator_upstream_part, collimator_upstream_positions[0], collimator_upstream_positions[2] + 29. / 2. * inch, collimator_upstream_base_theta * pi / 180.)
    
    with open('beam.py.in', 'w') as f_beam:
        f_beam.write('physics QGSP_BIC\n')
        f_beam.write('param worldMaterial=Air\n')
        f_beam.write('param histoFile=beam.root\n')
        f_beam.write('param inch=25.4\n')
    
        f_beam.write('g4ui when=4 "/vis/viewer/set/viewpointVector 0 1 0"\n')
        f_beam.write('g4ui when=4 "/vis/viewer/set/style wireframe"\n')
        # f_beam.write('g4ui when=4 "/vis/viewer/set/background 1 1 1"\n')
    
        f_beam.write('beam gaussian particle=pi+ firstEvent=0 lastEvent=1 sigmaX=2.0 sigmaY=2.0 beamZ=-500.0 meanMomentum=64000.0\n')
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
    
write()
