from ROOT import TDatabasePDG
from rootalias import *
from math import pi, cos, sin


PDG = TDatabasePDG()
f_beam = TFile('data/beam.1_spill.root')

z0 = 8005.9022 / 10.0
x0 = -1186.1546 / 10.0

def rotate_y(x, z, degree):
    theta = degree * pi / 180.0
    x = cos(theta) * x - sin(theta) * z
    z = sin(theta) * x + cos(theta) * z
    return x, z

tracks = []
for event in f_beam.Get('VirtualDetector/TOFds'):
    pid = int(event.PDGid)
    px = event.Px / 1000.0
    py = event.Py / 1000.0
    pz = event.Pz / 1000.0
    x = event.x / 10.0
    y = event.y / 10.0
    z = event.z / 10.0

    rotate_y_degree = -3.0
    px, pz = rotate_y(px, pz, rotate_y_degree)
    x, z = rotate_y(x - x0, z - z0, rotate_y_degree)

    mass = PDG.GetParticle(pid).Mass()
    energy = (mass**2 + px**2 + py**2 + pz**2)**0.5
    track = [
        1,
        pid,
        0,
        0,
        0,
        0,
        px,
        py,
        pz,
        energy,
        mass,
        x,
        y,
        z,
        event.t
    ]
    tracks.append(track)

with open ('beam.txt', 'w') as f_txt:
    f_txt.write('0 {}\n'.format(len(tracks)))
    for track in tracks:
        f_txt.write(' '.join(map(str, track)))
        f_txt.write('\n')
