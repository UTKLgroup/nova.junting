from math import cos, sin, pi

INCH = 25.4
KILL = 1

target_slab_dimensions = [31.75, 209.55, 6.35] # [height, length, width]
target_slab_positions = [0., 0., 0.]
target_slab_count = 5.
target_delta_x = target_slab_dimensions[0] / target_slab_count
target_delta_z = 22.145

collimator_upstream_base_dimensions = [5.19 * INCH, 58. * INCH, 32. * INCH]
collimator_upstream_bottom_dimensions = [5.19 / 2. * INCH, 42.76 * INCH, 32. * INCH]
collimator_upstream_middle_dimensions = [2. * INCH, 42.76 * INCH, 11.6 * INCH]
collimator_upstream_top_dimensions = [5.19 * INCH, 42.76 * INCH, 32. * INCH]

collimator_upstream_base_positions = [0., -6.19 * INCH, 0.]
collimator_upstream_bottom_positions = [0., -(1. + 5.19 / 4.) * INCH, 7.62 * INCH]
collimator_upstream_middle_1_positions = [296. / 2. + 67.29, 0., 7.62 * INCH]
collimator_upstream_middle_2_positions = [-296. / 2. - 67.29, 0., 7.62 * INCH]
collimator_upstream_top_positions = [0., (1. + 5.19 / 2.) * INCH, 7.62 * INCH]


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


collimator_upstream_parts = [
    collimator_upstream_base_positions,
    collimator_upstream_bottom_positions,
    collimator_upstream_middle_1_positions,
    collimator_upstream_middle_2_positions,
    collimator_upstream_top_positions
]

translation = [-8.315 * INCH + 40., 0., (29. / 2. + 7.62) * INCH]
collimator_upstream_theta = 3               # degree, positive here means a counter-clockwise rotation in the top view
for collimator_upstream_part in collimator_upstream_parts:
    translate(collimator_upstream_part, translation)
    rotate_y(collimator_upstream_part, translation[0], translation[2] + 29. / 2. * INCH, collimator_upstream_theta * pi / 180.)

with open('beam.py.in', 'w') as f_beam:
    f_beam.write('physics QGSP_BIC\n')
    f_beam.write('param worldMaterial=Air\n')
    f_beam.write('param histoFile=beam.root\n')
    f_beam.write('box slab height={} length={} width={} material=Cu color=1,0.01,0.01\n'.format(target_slab_dimensions[0], target_slab_dimensions[1], target_slab_dimensions[2]))
    for i in range(-2, 3):
        f_beam.write('place slab rename=target_slab_{} x={} z={}\n'.format(i, i * target_delta_x, -i * target_delta_z))

    f_beam.write('box collimator_upstream_base height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_base_dimensions[0], collimator_upstream_base_dimensions[1], collimator_upstream_base_dimensions[2], KILL))
    f_beam.write('box collimator_upstream_bottom height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_bottom_dimensions[0], collimator_upstream_bottom_dimensions[1], collimator_upstream_bottom_dimensions[2], KILL))
    f_beam.write('box collimator_upstream_middle height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_middle_dimensions[0], collimator_upstream_middle_dimensions[1], collimator_upstream_middle_dimensions[2], KILL))
    f_beam.write('box collimator_upstream_top height={} length={} width={} material=Fe color=0,1,1 kill={}\n'.format(collimator_upstream_top_dimensions[0], collimator_upstream_top_dimensions[1], collimator_upstream_top_dimensions[2], KILL))

    f_beam.write('place collimator_upstream_base rename=collimator_upstream_base x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_base_positions[0], collimator_upstream_base_positions[1], collimator_upstream_base_positions[2], collimator_upstream_theta))
    f_beam.write('place collimator_upstream_bottom rename=collimator_upstream_bottom x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_bottom_positions[0], collimator_upstream_bottom_positions[1], collimator_upstream_bottom_positions[2], collimator_upstream_theta))
    f_beam.write('place collimator_upstream_middle rename=collimator_upstream_middle_1 x={} y={} z={} rotation=y{}-14.03\n'.format(collimator_upstream_middle_1_positions[0], collimator_upstream_middle_1_positions[1], collimator_upstream_middle_1_positions[2], collimator_upstream_theta))
    f_beam.write('place collimator_upstream_middle rename=collimator_upstream_middle_2 x={} y={} z={} rotation=y{}-17.97\n'.format(collimator_upstream_middle_2_positions[0], collimator_upstream_middle_2_positions[1], collimator_upstream_middle_2_positions[2], collimator_upstream_theta))
    f_beam.write('place collimator_upstream_top rename=collimator_upstream_top x={} y={} z={} rotation=y{}\n'.format(collimator_upstream_top_positions[0], collimator_upstream_top_positions[1], collimator_upstream_top_positions[2], collimator_upstream_theta))
