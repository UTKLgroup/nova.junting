from rootalias import *
from pprint import pprint
import csv


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


# get_particle_count_filter()
# get_particle_count()
print_particle_count_table()
