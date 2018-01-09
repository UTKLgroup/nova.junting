from rootalias import *
from pprint import pprint
import csv
from math import pi, cos, sin


figure_dir = 'figures'


def get_particle_filter():
    tf1 = TFile('MergedAtStartLinebeam.1_spill.root')
    track_count = 0
    particles = []

    for track in tf1.Get('EventTree_Spill1'):
        track_count += 1
        pass_all = track.TrackPresentStartLine and track.TrackPresentTOFus and track.TrackPresentDet1 and track.TrackPresentDet2 and track.TrackPresentDet3 and track.TrackPresentDet4 and track.TrackPresentTOFds

        if pass_all:
            print('passed')
            particle = [track.EventID, track.TrackID, track.TrackPresentTOFds, track.xTOFds, track.yTOFds, track.zTOFds, track.tTOFds, track.PxTOFds, track.PyTOFds, track.PzTOFds, track.PDGidTOFds, track.ParentIDTOFds]
            particles.append(particle)

        if track_count % 1000000 == 0:
            print(track_count)

    with open('fraction.csv', 'w') as f_fraction:
        for particle in particles:
            particle = list(map(str, particle))
            f_fraction.write('{}\n'.format(','.join(particle)))

    pprint(particles)


def get_particle_count_filter(filename):
    particle_count = {}
    with open(filename) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            pdg = int(float(row[-2]))
            if pdg not in particle_count:
                particle_count[pdg] = 1
            else:
                particle_count[pdg] += 1
    return particle_count


def get_particle_count():
    tf1 = TFile('beam.1_spill.root')
    particle_count = {}
    for track in tf1.Get('VirtualDetector/TOFds'):
        pdg = int(track.PDGid)
        if pdg not in particle_count:
            particle_count[pdg] = 1
        else:
            particle_count[pdg] += 1
    return particle_count


def print_particle_count_table():
    particle_count = get_particle_count()
    particle_count_filter_all = get_particle_count_filter('fraction.all.csv')
    particle_count_filter_tof = get_particle_count_filter('fraction.tof.csv')
    pdg = TDatabasePDG()
    for pdg_id in particle_count.keys():
        name = pdg.GetParticle(pdg_id).GetName()
        print('{} & {} & {} & {} & {} \\\\'.format(pdg_id, name, particle_count.get(pdg_id, ''), particle_count_filter_tof.get(pdg_id, ''), particle_count_filter_all.get(pdg_id, '')))


def get_csv(filename):
    particles = []
    with open(filename) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            particles.append({
                'EventID': int(row[0]),
                'TrackID': int(row[1]),
                'TrackPresent': int(row[2]),
                'x': float(row[3]),
                'y': float(row[4]),
                'z': float(row[5]),
                't': float(row[6]),
                'Px': float(row[7]),
                'Py': float(row[8]),
                'Pz': float(row[9]),
                'PDGid': int(float(row[10])),
                'ParentID': int(float(row[11]))
            })
    particles = sorted(particles, key=lambda x: x['t'])
    return particles


def rotate_y(x, z, degree):
    theta = degree * pi / 180.0
    x = cos(theta) * x - sin(theta) * z
    z = sin(theta) * x + cos(theta) * z
    return x, z


def generate_text():
    particles = get_csv('fraction.tof.csv')
    PDG = TDatabasePDG()

    z0 = 8005.9022 / 10.0
    x0 = -1186.1546 / 10.0
    t0 = 60.
    delta_t = 0.55

    h1 = TH1D('h1', 'h1', 300, 58, 66)
    with open ('beam.txt', 'w') as f_txt:
        tracks = []
        event_count = 0
        particle_count = 0
        for i, particle in enumerate(particles):
            pid = particle['PDGid']
            px = particle['Px'] / 1000.0
            py = particle['Py'] / 1000.0
            pz = particle['Pz'] / 1000.0
            x = particle['x'] / 10.0
            y = particle['y'] / 10.0
            z = particle['z'] / 10.0

            rotate_y_degree = -3.0
            px, pz = rotate_y(px, pz, rotate_y_degree)
            x, z = rotate_y(x - x0, z - z0, rotate_y_degree)
            t = particle['t']
            mass = PDG.GetParticle(pid).Mass()
            energy = (mass**2 + px**2 + py**2 + pz**2)**0.5
            h1.Fill(particle['t'])

            if t > t0 + delta_t:
                f_txt.write('0 {}\n'.format(len(tracks)))
                for track in tracks:
                    track[-1] = (track[-1] - t0) * 1.e9
                    f_txt.write(' '.join(map(str, track)) + '\n')
                    particle_count += 1
                tracks = []
                t0 += delta_t
                event_count += 1
            tracks.append([1, pid, 0, 0, 0, 0, px, py, pz, energy, mass, x, y, z, t])

        f_txt.write('0 {}\n'.format(len(tracks)))
        for track in tracks:
            track[-1] = (track[-1] - t0) * 1.e9
            f_txt.write(' '.join(map(str, track)) + '\n')
            particle_count += 1
        event_count += 1

    print('event_count = ', event_count)
    print('particle_count = ', particle_count)
    print('len(particles) = ', len(particles))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    h1.Draw()

    c1.Update()
    c1.SaveAs('{}/generate_text.pdf'.format(figure_dir))
    input('Press any key to continue.')


# 20171211_test_beam_geometry
# get_particle_count_filter()
# get_particle_count()
# print_particle_count_table()
generate_text()
