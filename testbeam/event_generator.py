from ROOT import TDatabasePDG, TFile, gDirectory, TH1D, TH2D
from math import pi, cos, sin
import argparse
import os


class EventGenerator:
    EVENT_TIME_DURATION = 50.e3  # ns, 50 us per event

    def __init__(self, filename, exclude_noise, save_plot, max_spill_count):
        self.filename = filename
        self.exclude_noise = exclude_noise
        self.save_plot = save_plot
        self.data_dir = os.path.dirname(self.filename)
        if not self.data_dir:
            self.data_dir = '.'
        self.file_basename = os.path.basename(self.filename)
        self.max_spill_count = max_spill_count

        self.pdg = TDatabasePDG()
        self.delta_x = 1.4  # mm
        self.delta_y = 8.6  # mm
        self.delta_z = 0.  # mm
        # self.delta_x = 1375.9  # mm
        # self.delta_y = -67.5  # mm
        # self.delta_z = -14617.4  # mm
        self.angle_rotation_y_axis = -0.349 * pi / 180.  # rad

    def rotate_y_axis(self, z, x):
        # angle > 0 if the coordinate system rotates counterclockwise
        # angle < 0 if the coordinate system rotates clockwise
        z_prime = cos(self.angle_rotation_y_axis) * z + sin(self.angle_rotation_y_axis) * x
        x_prime = -sin(self.angle_rotation_y_axis) * z + cos(self.angle_rotation_y_axis) * x
        return z_prime, x_prime

    def translate(self, x, y, z):
        return x + self.delta_x, y + self.delta_y, z + self.delta_z

    def run(self):
        particles = []
        particle_count = 0
        spill_count = 0

        f_beam = TFile(self.filename)
        spills = [key.GetName() for key in gDirectory.GetListOfKeys()]
        for spill in spills:
            spill_count += 1
            if self.max_spill_count and spill_count > self.max_spill_count:
                spill_count -= 1
                break
            print('spill = {}'.format(spill))

            for track in f_beam.Get(spill):
                particle_count += 1
                pass_all = track.TrackPresenttof_us and \
                    track.TrackPresentwire_chamber_1_detector and \
                    track.TrackPresentwire_chamber_2_detector and \
                    track.TrackPresentwire_chamber_3_detector and \
                    track.TrackPresentwire_chamber_4_detector and \
                    track.TrackPresentcherenkov and \
                    track.TrackPresenttof_ds and \
                    track.TrackPresentnova

                is_noise = not pass_all
                if self.exclude_noise and is_noise is True:
                    continue

                pdg_id = int(track.PDGidnova)
                x, y, z = self.translate(track.xnova, track.ynova, track.znova)  # mm
                # z, x = self.rotate_y_axis(z, x)
                t = track.tnova  # s

                px = track.Pxnova  # MeV
                py = track.Pynova  # MeV
                pz = track.Pznova  # MeV
                # pz, px = self.rotate_y_axis(pz, px)

                px /= 1000.  # GeV
                py /= 1000.  # GeV
                pz /= 1000.  # GeV
                x /= 10.  # cm
                y /= 10.  # cm
                z /= 10.  # cm
                t *= 1.e9  # ns

                mass = self.pdg.GetParticle(pdg_id).Mass()
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
                particle_count += 1
                if particle_count % 1e6 == 0:
                    print('particle_count = {}'.format(particle_count))

        print('spill_count = {}'.format(spill_count))
        f_beam.Close()

        particles = sorted(particles, key=lambda p: p[-1])
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
                    event_end_time = time + EventGenerator.EVENT_TIME_DURATION  # ns
            else:
                particle[-1] -= event_start_time
                event_particles.append(particle)
        if event_particles:
            events.append(event_particles)

        txt_filename = 'text_gen.{}{}.txt'.format(self.file_basename, '.exclude_noise' if self.exclude_noise else '')
        with open('{}/{}'.format(self.data_dir, txt_filename), 'w') as f_txt:
            for event in events:
                f_txt.write('0 {}\n'.format(len(event)))
                for particle in event:
                    f_txt.write(' '.join(map(str, particle[1:])) + '\n')

        if self.save_plot:
            self.make_plot(events)

    def make_plot(self, events):
            f_det = TFile('{}/text_gen.{}{}.root'.format(self.data_dir, self.file_basename, '.exclude_noise' if self.exclude_noise else ''), 'RECREATE')
            multiple_particle_event_count = 0
            h_count = TH1D('h_count', 'h_count', 100, -0.5, 99.5)
            h_timing = TH1D('h_timing', 'h_timing', 5000, 0., 50.e3)  # ns
            h_xy = TH2D('h_xy', 'h_xy', 600, -150, 150, 600, -150, 150)  # cm
            h_z = TH1D('h_z', 'h_z', 500, -1, 1)  # cm
            for i, event in enumerate(events):
                h_count.Fill(len(event))
                if len(event) > 1:
                    multiple_particle_event_count += 1

                for particle in event:
                    h_timing.Fill(particle[-1])
                    is_noise = particle[0]
                    if is_noise:
                        continue
                    x = particle[-4]
                    y = particle[-3]
                    z = particle[-2]
                    h_xy.Fill(x, y)
                    h_z.Fill(z)

            print('len(events) = {}'.format(len(events)))
            print('multiple_particle_event_count = {}'.format(multiple_particle_event_count))
            h_count.Write('h_particle_count_per_event')
            h_timing.Write('h_timing')
            h_xy.Write('h_xy')
            h_z.Write('h_z')
            f_det.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='input filename', required=True)
    parser.add_argument('-s', '--save_plot', help='save particle count per event', action='store_true')
    parser.add_argument('-e', '--exclude_noise', help='exclude noise', action='store_true')
    parser.add_argument('-m', '--max_spill_count', type=int, help='max spill count', default=None)

    args = parser.parse_args()
    print('args.filename = {}'.format(args.filename))
    print('args.save_plot = {}'.format(args.save_plot))
    print('args.exclude_noise = {}'.format(args.exclude_noise))
    print('args.max_spill_count = {}'.format(args.max_spill_count))

    event_generator = EventGenerator(args.filename, args.exclude_noise, args.save_plot, args.max_spill_count)
    event_generator.run()
