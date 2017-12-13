from rootalias import *
from pprint import pprint


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
