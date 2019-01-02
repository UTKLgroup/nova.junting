from ROOT import TDatabasePDG
from rootalias import *
from math import pi, cos, sin
from subprocess import call
import argparse
import os


PDG = TDatabasePDG()
EVENT_TIME_DURATION = 50.e3         # ns, 50 us per event


def convert_to_detsim_coordinates(beamsim_x, beamsim_y, beamsim_z):
    return beamsim_x + 1354.35596, beamsim_y, beamsim_z - 14648.66998


def rotate_y(x, z, degree):
    theta = degree * pi / 180.0
    x = cos(theta) * x - sin(theta) * z
    z = sin(theta) * x + cos(theta) * z
    return x, z


def save_to_txt(filename, include_noise, save_plot):
    f_beam = TFile(filename)
    particles = []
    tree = f_beam.Get('tree')
    particle_count_total = tree.GetEntries()
    particle_count = 0

    for event in f_beam.Get('tree'):
        particle_count += 1
        if particle_count % 1e6 == 0:
            print('particle_count = {} / {} ({:.1f}%)'.format(particle_count, particle_count_total, particle_count / particle_count_total * 100.))

        if not include_noise and event.is_noise == 1:
            continue

        is_noise = event.is_noise
        pdg_id = int(event.pdg_id)
        px = event.px / 1000.   # GeV
        py = event.py / 1000.
        pz = event.pz / 1000.
        x, y, z = convert_to_detsim_coordinates(event.x, event.y, event.z)
        x /= 10.                # cm
        y /= 10.
        z /= 10.
        t = event.t * 1.e9      # ns

        mass = PDG.GetParticle(pdg_id).Mass()
        energy = (mass**2 + px**2 + py**2 + pz**2)**0.5
        particle = [
            is_noise,
            1, pdg_id,
            0, 0, 0, 0,
            px, py, pz,
            energy, mass,
            x, y, z, t
        ]
        particles.append(particle)
    f_beam.Close()

    particles = sorted(particles, key=lambda x: x[-1])
    events = []
    event_end_time = 0.
    event_start_time = 0.
    event_particles = []
    for particle in particles:
        is_noise = particle[0]
        time = particle[-1]     # time
        if time > event_end_time:
            if event_particles:
                events.append(event_particles)
                event_particles = []
            if not is_noise:
                particle[-1] = 0.
                event_particles = [particle]
                event_start_time = time
                event_end_time = time + EVENT_TIME_DURATION # ns
        else:
            particle[-1] -= event_start_time
            event_particles.append(particle)

    data_dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    txt_filename = 'text_gen.{}.txt'.format(filename)
    with open ('{}/{}'.format(data_dir, txt_filename), 'w') as f_txt:
        for event in events:
            f_txt.write('0 {}\n'.format(len(event)))
            for particle in event:
                particle.pop(0)
                f_txt.write(' '.join(map(str, particle)) + '\n')
    # call('scp {}/{} junting@novagpvm02.fnal.gov:/nova/app/users/junting/testbeam/det/'.format(data_dir, txt_filename), shell=True)

    if save_plot:
        f_det = TFile('{}/text_gen.{}.root'.format(data_dir, filename), 'RECREATE')
        multiple_particle_event_count = 0
        h1 = TH1D('h1', 'h1', 100, -0.5, 99.5)
        for i, event in enumerate(events):
            h1.Fill(len(event))
            if len(event) > 2:
                multiple_particle_event_count += 1
                # print('i = {}'.format(i))
                # print('len(event) = {}'.format(len(event)))
        print('len(events) = {}'.format(len(events)))
        print('multiple_particle_event_count = {}'.format(multiple_particle_event_count))
        h1.Write('h_particle_count_per_event')
        f_det.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='input filename', required=True)
    parser.add_argument('-s', '--save_plot', help='save particle count per event', action='store_true')
    parser.add_argument('-n', '--include_noise', help='include noise', action='store_true')

    args = parser.parse_args()
    print('args.filename = {}'.format(args.filename))
    print('args.save_plot = {}'.format(args.save_plot))
    print('args.include_noise = {}'.format(args.include_noise))
    save_to_txt(args.filename, args.include_noise, args.save_plot)
