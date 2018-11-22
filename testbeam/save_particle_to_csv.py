from ROOT import TFile, gDirectory
import argparse
import os
from pprint import pprint


def save_particle_to_csv(filename):
    tf1 = TFile(filename)

    pid_momentums = {}
    particles = []
    noise_particles = []

    keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
    for key in keys:
        print('key = {}'.format(key))
        track_count = 0
        for track in tf1.Get(key):
            track_count += 1
            # pass_all = track.TrackPresentstart_line and \
            #            track.TrackPresenttof_us and \
            pass_all = track.TrackPresenttof_us and \
                       track.TrackPresentwire_chamber_1_detector and \
                       track.TrackPresentwire_chamber_2_detector and \
                       track.TrackPresentwire_chamber_3_detector and \
                       track.TrackPresentwire_chamber_4_detector and \
                       track.TrackPresenttof_ds and \
                       track.TrackPresentcherenkov and \
                       track.TrackPresentnova

            if track_count % 100000 == 0:
                print('track_count = {}'.format(track_count))

            if track.TrackPresentnova:
                particle = [
                    track.EventID, track.TrackID,
                    track.ttof_us, track.ttof_ds,
                    track.xnova, track.ynova, track.znova, track.tnova, track.Pxnova, track.Pynova, track.Pznova, track.PDGidnova, track.ParentIDnova
                ]

                if pass_all:
                    print('passed!')
                    particles.append(particle)

                    pid = track.PDGidtof_ds
                    momentum = (track.Pxtof_ds**2 + track.Pytof_ds**2 + track.Pztof_ds**2)**0.5
                    if pid not in pid_momentums:
                        pid_momentums[pid] = [momentum]
                    else:
                        pid_momentums[pid].append(momentum)
                    print('track.PDGidtof_ds = {}'.format(track.PDGidtof_ds))
                    print('momentum = {}'.format(momentum))
                else:
                    noise_particles.append(particle)

    with open('{}.csv'.format(os.path.basename(filename)), 'w') as f_fraction:
        for particle in particles:
            f_fraction.write('0,{}\n'.format(','.join(list(map(str, particle)))))
        for noise_particle in noise_particles:
            f_fraction.write('1,{}\n'.format(','.join(list(map(str, noise_particle)))))

    pprint(pid_momentums)


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    print('processing {}'.format(args.filename))
    save_particle_to_csv(args.filename)
