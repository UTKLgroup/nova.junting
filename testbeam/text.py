from ROOT import TDatabasePDG
from rootalias import *
from math import pi, cos, sin
from subprocess import call


DATA_DIR = './data'
FIGURE_DIR = '/Users/juntinghuang/beamer/20181203_testbeam_bridge_beam_detsim/figures'
PDG = TDatabasePDG()


def convert_to_detsim_coordinates(beamsim_x, beamsim_y, beamsim_z):
    return beamsim_x + 1354.35596, beamsim_y, beamsim_z - 14648.66998


def rotate_y(x, z, degree):
    theta = degree * pi / 180.0
    x = cos(theta) * x - sin(theta) * z
    z = sin(theta) * x + cos(theta) * z
    return x, z


def save_to_txt(filename):
    event_time_duration = 50.e3         # ns, 50 us per event

    f_beam = TFile('{}/{}'.format(DATA_DIR, filename))
    particles = []
    for event in f_beam.Get('tree'):
        if event.is_noise == 1:
            continue

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
            1, pdg_id,
            0, 0, 0, 0,
            px, py, pz,
            energy, mass,
            x, y, z, t
        ]
        particles.append(particle)

    particles = sorted(particles, key=lambda x: x[-1])
    events = []
    event_end_time = 0.
    event_start_time = 0.
    event_particles = []
    for particle in particles:
        time = particle[-1]     # time
        if time > event_end_time:
            if event_particles:
                events.append(event_particles)
            particle[-1] = 0.
            event_particles = [particle]
            event_start_time = time
            event_end_time = time + event_time_duration # ns
        else:
            particle[-1] -= event_start_time
            event_particles.append(particle)

    txt_filename = 'text_gen.{}.txt'.format(filename)
    with open ('{}/{}'.format(DATA_DIR, txt_filename), 'w') as f_txt:
        for event in events:
            f_txt.write('0 {}\n'.format(len(event)))
            for particle in event:
                f_txt.write(' '.join(map(str, particle)) + '\n')
    # call('scp {}/{} junting@novagpvm02.fnal.gov:/nova/app/users/junting/testbeam/det/'.format(DATA_DIR, txt_filename), shell=True)

    event_count = len(events)
    multiple_particle_event_count = 0
    h1 = TH1D('h1', 'h1', 5, -0.5, 4.5)
    for i, event in enumerate(events):
        h1.Fill(len(event))
        if len(event) > 2:
            multiple_particle_event_count += 1
            print('i = {}'.format(i))
            print('len(event) = {}'.format(len(event)))

    print('event_count = {}'.format(event_count))
    print('multiple_particle_event_count = {}'.format(multiple_particle_event_count))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()
    set_h1_style(h1)
    h1.GetXaxis().SetTitle('Particle Count per Event')
    h1.GetYaxis().SetTitle('Event Count')
    h1.GetYaxis().SetMaxDigits(3)
    h1.Draw()
    c1.Update()
    c1.SaveAs('{}/save_to_txt.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


if __name__ == '__main__':
    gStyle.SetOptStat(0)
    # save_to_txt('g4bl.b_-0.9T.proton.64000.root.job_30000_32000.40m.kineticEnergyCut_20.root')
    save_to_txt('g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root')
