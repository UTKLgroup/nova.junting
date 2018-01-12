from rootalias import *
from pprint import pprint
import csv
from math import pi, cos, sin
import numpy as np


PDG = TDatabasePDG()
SPEED_OF_LIGHT = 3.e8           # m/s
FIGURE_DIR = '/Users/juntinghuang/beamer/20180109_testbeam_momentum_pid/figures'


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

    rotate_y_degree = -3.0
    z0 = 8005.9022 / 10.0
    x0 = -1186.1546 / 10.0
    t0 = 60.
    delta_t = 550.e-6

    event_id_particle_ids = {}
    h1 = TH1D('h1', 'h1', 300, 0, 5.)
    for i, particle in enumerate(particles):
        t = particle['t'] - t0
        h1.Fill(t)
        event_id = int(t // delta_t)
        if event_id not in event_id_particle_ids:
            event_id_particle_ids[event_id] = [i]
        else:
            event_id_particle_ids[event_id].append(i)

    pprint(event_id_particle_ids)
    for event_id, particle_ids in event_id_particle_ids.items():
        particle_count = len(particle_ids)
        if particle_count > 1:
            print('pile up of {} particles in event {} '.format(paticle_count, event_id))


    with open ('beam.txt', 'w') as f_txt:
        particle_count = 0
        for event_id, particle_ids in event_id_particle_ids.items():
            f_txt.write('0 {}\n'.format(len(particle_ids)))
            for particle_id in particle_ids:
                particle = particles[particle_id]
                pid = particle['PDGid']
                px = particle['Px'] / 1000.0
                py = particle['Py'] / 1000.0
                pz = particle['Pz'] / 1000.0
                x = particle['x'] / 10.0
                y = particle['y'] / 10.0
                z = particle['z'] / 10.0

                px, pz = rotate_y(px, pz, rotate_y_degree)
                x, z = rotate_y(x - x0, z - z0, rotate_y_degree)
                t = (particle['t'] - event_id * delta_t - t0) * 1.e9
                mass = PDG.GetParticle(pid).Mass()
                energy = (mass**2 + px**2 + py**2 + pz**2)**0.5

                track = [1, pid, 0, 0, 0, 0, px, py, pz, energy, mass, x, y, z, t]
                f_txt.write(' '.join(map(str, track)) + '\n')
                particle_count += 1

    print('event_count = ', len(event_id_particle_ids))
    print('particle_count = ', particle_count)
    print('len(particles) = ', len(particles))

    gStyle.SetOptStat(0)
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)
    h1.Draw()
    h1.GetXaxis().SetTitle('Time (s)')
    h1.GetYaxis().SetTitle('Particle Count')

    # t0s = []
    # for i in range(1, event_count + 1):
    #     t0s.append(delta_t * i)
    # tls = []
    # c1.Update()
    # for i, t0 in enumerate(t0s):
    #     tl = TLine(t0, gPad.GetUymin(), t0, gPad.GetUymax())
    #     tl.SetLineColor(kRed + 1)
    #     tl.SetLineStyle(7)
    #     tl.SetLineWidth(2)
    #     tls.append(tl)
    #     tls[i].Draw()

    c1.Update()
    c1.SaveAs('{}/generate_text.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_momentum(kinetic_energy, mass):
    return (kinetic_energy**2 + 2. * mass * kinetic_energy)**0.5


def plot_momentum():
    particles = get_csv('fraction.tof.csv')
    h1 = TH1D('h1', 'h1', 100, 0, 1500)
    for particle in particles:
        p = (particle['Px']**2 + particle['Py']**2 + particle['Pz']**2)**0.5
        h1.Fill(p)
        if p < 200:
            print(particle['PDGid'])
        if particle['PDGid'] == 11:
            print(p)

    gStyle.SetOptStat('emr')
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)
    h1.Draw()
    h1.GetXaxis().SetTitle('P (MeV)')
    h1.GetYaxis().SetTitle('Particle Count')
    gPad.SetLogy()
    c1.Update()
    draw_statbox(h1, x1=0.7)

    c1.Update()
    c1.SaveAs('{}/plot_momentum.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_p_vs_angle():
    b_field = 0.14                   # tesla
    field_length = 591. / 1000. * 2. # m

    unit_charge = 1.602e-19     # coulomb
    joule_mev = 1. / unit_charge
    degree_to_radian = 3.14 / 180.

    degrees = np.arange(0.1, 15., 0.1)
    b_fields = [0.14, 0.175, 0.35]
    colors = [kBlue + 2, kGreen + 2, kRed + 2]

    b_field_momentums = []
    for b_field in b_fields:
        momentums = []
        for degree in degrees:
            momentum = b_field * field_length * SPEED_OF_LIGHT / (degree * degree_to_radian) * 1.e-6 # MeV
            momentums.append(momentum)
        b_field_momentums.append(momentums)

    print(degrees)
    print(momentums)

    gr_0 = TGraph(len(degrees), np.array(degrees), np.array(b_field_momentums[0]))
    gr_1 = TGraph(len(degrees), np.array(degrees), np.array(b_field_momentums[1]))
    gr_2 = TGraph(len(degrees), np.array(degrees), np.array(b_field_momentums[2]))

    gr_data_0 = TGraph(1, np.array([12.]), np.array([250.]))
    gr_data_1 = TGraph(1, np.array([10.]), np.array([450.]))
    gr_data_2 = TGraph(1, np.array([12.]), np.array([600.]))

    for gr in [gr_0, gr_1, gr_2, gr_data_0, gr_data_1, gr_data_2]:
        set_graph_style(gr)
        gr.SetMarkerStyle(24)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gr_0.Draw('AL')
    gr_0.GetYaxis().SetRangeUser(0., 3000.)
    gr_0.GetXaxis().SetTitle('Magnet Bending Angle (degree)')
    gr_0.GetYaxis().SetTitle('Particle Momentum (MeV)')
    gr_0.GetYaxis().SetTitleOffset(1.5)
    gr_0.SetLineColor(colors[0])
    gPad.SetGrid()
    gr_data_0.SetMarkerColor(colors[0])
    gr_data_0.Draw('sames,P')

    # gr_1.Draw('sames,L')
    # gr_1.SetLineColor(colors[1])
    # gr_data_1.SetMarkerColor(colors[1])
    # gr_data_1.Draw('sames,P')

    gr_2.Draw('sames,L')
    gr_2.SetLineColor(colors[2])
    gr_data_2.SetMarkerColor(colors[2])
    gr_data_2.Draw('sames,P')

    lg1 = TLegend(0.4, 0.7, 0.88, 0.88)
    set_legend_style(lg1)
    lg1.SetNColumns(2)

    lg1.AddEntry(gr_2, 'B = 0.35 T', 'l')
    lg1.AddEntry(gr_data_2, 'MC peak', 'p')
    lg1.AddEntry(gr_0, 'B = 0.14 T', 'l')
    lg1.AddEntry(gr_data_0, 'MC peak', 'p')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_p_vs_angle.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_cherenkov():
    names = ['proton', 'K+', 'pi+', 'mu+', 'e+']
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    colors = [kRed + 2, kMagenta + 2, kBlue + 2, kGreen + 2, kBlack]
    eta = 4.1e-4                  # atm-1

    momentums = np.arange(0.01, 10, 0.01)
    ppressures = []
    for i, mass in enumerate(masses):
        pressures = []
        for momentum in momentums:
            pressure = 1. / eta * ((1 + (mass / momentum)**2)**0.5 - 1.)
            pressures.append(pressure)
        ppressures.append(pressures)

    grs = []
    for i in range(len(ppressures)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(ppressures[i]))
        set_graph_style(gr)
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()
    gPad.SetGrid()

    lg1 = TLegend(0.2, 0.8, 0.88, 0.88)
    set_legend_style(lg1)
    lg1.SetNColumns(5)

    grs[0].Draw('AL')
    grs[0].SetLineColor(colors[0])
    grs[0].GetXaxis().SetRangeUser(0., 3)
    grs[0].GetYaxis().SetRangeUser(1.e-5, 1e6)
    grs[0].GetYaxis().SetTitle('Pressure Threshold (atm)')
    grs[0].GetXaxis().SetTitle('Momentum (GeV)')
    lg1.AddEntry(grs[0], names[0], 'l')
    for i in range(1, len(names)):
        grs[i].Draw('sames,L')
        grs[i].SetLineColor(colors[i])
        lg1.AddEntry(grs[i], names[i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_cherenkov.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_time_of_flight():
    distance = 10.            # m
    names = ['proton', 'K+', 'pi+', 'mu+', 'e+']
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    colors = [kRed + 2, kMagenta + 2, kBlue + 2, kGreen + 2, kBlack]

    momentums = np.arange(0.001, 10, 0.001)
    ttofs = []
    for i in range(len(names)):
        tofs= []
        for momentum in momentums:
            tof = distance / SPEED_OF_LIGHT * (1. + (masses[i] / momentum)**2)**0.5 * 1.e9
            tofs.append(tof)
        ttofs.append(tofs)

    grs = []
    for i in range(len(ttofs)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(ttofs[i]))
        gr.SetLineColor(colors[i])
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()
    gPad.SetLogx()


    lg1 = TLegend(0.65, 0.5, 0.9, 0.85)
    set_legend_style(lg1)

    grs[0].Draw('AL')
    grs[0].GetXaxis().SetRangeUser(0, 3)
    grs[0].GetXaxis().SetTitle('Momentum (GeV)')
    grs[0].GetYaxis().SetTitle('Time of Flight (ns)')
    grs[0].GetYaxis().SetRangeUser(10., 1.e5)
    set_graph_style(grs[0])
    lg1.AddEntry(grs[0], names[0], 'l')

    for i in range(1, len(names)):
        set_graph_style(grs[i])
        grs[i].Draw('sames,L')
        lg1.AddEntry(grs[i], names[i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_time_of_flight.pdf'.format(figure_dir))
    input('Press any key to continue.')


# 20180109_testbeam_momentum_pid
# plot_p_vs_angle()
# plot_cherenkov()
plot_time_of_flight()


# 20171211_test_beam_geometry
# get_particle_count_filter()
# get_particle_count()
# print_particle_count_table()
# generate_text()
# print(get_momentum(237.843, 938.272))
# plot_momentum()
