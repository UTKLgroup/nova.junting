from rootalias import *
from pprint import pprint
import csv
import math
from math import pi, cos, sin, atan, sqrt, log
import numpy as np


PDG = TDatabasePDG()
SPEED_OF_LIGHT = 3.e8              # m/s
ELEMENTARY_CHARGE = 1.60217662e-19 # coulomb
INCH_TO_METER = 2.54 / 100.
DEGREE_TO_RADIAN = 3.14 / 180.
RADIAN_TO_DEGREE = 180. / 3.14
FIGURE_DIR = '/Users/juntinghuang/beamer/20180625_testbeam_64_32_16_8GeV_different_bs/figures'
DATA_DIR = './data'


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


def plot_time_of_flight(**kwargs):
    # distance = 6.075            # m
    distance = kwargs.get('distance', 6.075) # m
    y_min = kwargs.get('y_min', 9.9e3)
    y_max = kwargs.get('y_max', 2.e5)
    canvas_height = kwargs.get('canvas_height', 800)

    names = ['proton', 'K+', 'pi+', 'mu+', 'e+']
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    colors = [kRed + 2, kMagenta + 2, kBlue + 2, kGreen + 2, kBlack]

    momentums = np.arange(0.001, 10, 0.001)
    ttofs = []
    for i in range(len(names)):
        tofs= []
        for momentum in momentums:
            tof = distance / SPEED_OF_LIGHT * (1. + (masses[i] / momentum)**2)**0.5 * 1.e12
            tofs.append(tof)
        ttofs.append(tofs)

    grs = []
    for i in range(len(ttofs)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(ttofs[i]))
        gr.SetLineColor(colors[i])
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, canvas_height)
    set_margin()
    # gPad.SetLogx()
    gPad.SetLogy()
    gPad.SetGrid()

    lg1 = TLegend(0.65, 0.5, 0.9, 0.85)
    set_legend_style(lg1)

    set_graph_style(grs[0])
    grs[0].Draw('AL')
    grs[0].GetXaxis().SetRangeUser(0, 3)
    grs[0].GetXaxis().SetTitle('Momentum (GeV)')
    grs[0].GetYaxis().SetTitle('Time of Flight (ps)')
    grs[0].GetYaxis().SetRangeUser(y_min, y_max)
    grs[0].GetXaxis().SetRangeUser(1.e-1, 3.)
    # grs[0].GetYaxis().SetTitleOffset(1.8)
    lg1.AddEntry(grs[0], names[0], 'l')

    for i in range(1, len(names)):
        set_graph_style(grs[i])
        grs[i].Draw('sames,L')
        lg1.AddEntry(grs[i], names[i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_time_of_flight.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_time_of_flight_diff(**kwargs):
    # distance = 6.075          # m
    # distance = 9.1            # m
    # distance = 12.            # m
    distance = kwargs.get('distance', 6.075) # m
    y_min = kwargs.get('y_min', 10.)
    y_max = kwargs.get('y_max', 1.e6)
    canvas_height = kwargs.get('canvas_height', 800)

    names = ['proton', 'K+', 'pi+', 'mu+', 'e+']
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    colors = [kRed + 2, kMagenta + 2, kBlue + 2, kGreen + 2, kBlack]
    name_diffs = ['proton - K+', 'K+ - pi+', 'pi+ - mu+', 'mu+ - e+']

    momentums = np.arange(0.001, 10, 0.001)
    ttofs = []
    for i in range(len(names)):
        tofs= []
        for momentum in momentums:
            tof = distance / SPEED_OF_LIGHT * (1. + (masses[i] / momentum)**2)**0.5 * 1.e12
            # tof /= 35.
            tofs.append(tof)
        ttofs.append(tofs)

    ttof_diffs = []
    for i in range(len(ttofs) - 1):
        tof_diffs = []
        for j in range(len(ttofs[i])):
            tof_diff = ttofs[i][j] - ttofs[i + 1][j]
            tof_diffs.append(tof_diff)
        ttof_diffs.append(tof_diffs)

    grs = []
    for i in range(len(ttof_diffs)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(ttof_diffs[i]))
        gr.SetLineColor(colors[i])
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, canvas_height)
    set_margin()
    gPad.SetLogy()
    gPad.SetGrid()

    # lg1 = TLegend(0.5, 0.6, 0.85, 0.9)
    lg1 = TLegend(0.5, 0.56, 0.85, 0.86)
    set_legend_style(lg1)

    set_graph_style(grs[0])
    grs[0].Draw('AL')
    grs[0].GetXaxis().SetTitle('Momentum (GeV)')
    grs[0].GetYaxis().SetTitle('Time of Flight (ps)')
    grs[0].GetYaxis().SetRangeUser(y_min, y_max)
    grs[0].GetXaxis().SetRangeUser(0., 3.)
    # grs[0].GetYaxis().SetTitleOffset(1.8)
    lg1.AddEntry(grs[0], name_diffs[0], 'l')

    for i in range(1, len(grs)):
        set_graph_style(grs[i])
        grs[i].Draw('sames,L')
        lg1.AddEntry(grs[i], name_diffs[i], 'l')

    tl = TLine(0., 200., 3., 200.)
    tl.SetLineWidth(3)
    tl.SetLineStyle(7)
    tl.Draw('sames')
    lg1.AddEntry(tl, 'timing resolution', 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_time_of_flight_diff.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def compute_bending_angle():
    b_field = 1.8
    b_field_length = 42. * 2.54 / 100.

    tf = TFile('magnet.root')
    for event in tf.Get('VirtualDetector/Detector'):
        theta = atan(event.Px / event.Pz)
        theta_degree = theta * 180. / pi
        momentum = (event.Px**2 + event.Pz**2)**0.5 / 1.e3      # GeV
        momentum_reconstruct = b_field * b_field_length / theta # si unit
        momentum_reconstruct = momentum_reconstruct * SPEED_OF_LIGHT / 1.e9 # GeV
        print('momentum = ', momentum)
        print('momentum_reconstruct = ', momentum_reconstruct)
        print('theta = ', theta)
        print('theta_degree = ', theta_degree)
        break


def compute_b_times_l():
    momentum_gev = 2.           # GeV
    theta_degree = 16.          # degree

    momentum_si = momentum_gev * 1.e9 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT
    theta_radian = theta_degree * pi / 180.

    bl = momentum_si / ELEMENTARY_CHARGE * theta_radian
    print('bl = ', bl)


def get_min_momentum(**kwargs):
    b_field = kwargs.get('b_field', 1.8) # T
    b_field_length = kwargs.get('b_field_length', 42. * INCH_TO_METER)
    aperture_width = kwargs.get('aperture_width', 4. * INCH_TO_METER)

    sagitta = aperture_width
    half_chord = b_field_length

    min_radius = sagitta / 2. + half_chord**2 / (2. * sagitta)
    min_momentum_si = ELEMENTARY_CHARGE * min_radius * b_field
    min_momentum_gev = min_momentum_si * SPEED_OF_LIGHT / ELEMENTARY_CHARGE / 1.e9
    max_theta_radian = half_chord / min_radius
    max_theta_degree = max_theta_radian * 180. / pi

    print('min_momentum_gev = ', min_momentum_gev)
    print('max_theta_degree = ', max_theta_degree)

    return min_momentum_gev, max_theta_degree


def get_max_theta(**kwargs):
    b_field_length = kwargs.get('b_field_length', 42. * INCH_TO_METER)
    aperture_width = kwargs.get('aperture_width', 4. * INCH_TO_METER)

    sagitta = aperture_width
    half_chord = b_field_length

    min_radius = sagitta / 2. + half_chord**2 / (2. * sagitta)
    max_theta_radian = half_chord / min_radius
    max_theta_degree = max_theta_radian * 180. / pi

    return max_theta_radian, max_theta_degree


def plot_m1_downstream():
    h1 = TH2D('h1', 'h1', 900, 0., 90., 600, 0., 15.)

    tf = TFile('magnet.root')
    event_count = 0
    for event in tf.Get('VirtualDetector/DownstreamDetector'):
        theta = abs(atan(event.Px / event.Pz))
        theta_degree = theta * 180. / pi
        momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1.e3      # GeV
        h1.Fill(theta_degree, momentum)
        event_count += 1
        # if event_count == 1000:
            # break

    b_field = 1.8
    field_length = 42. * INCH_TO_METER
    degrees = np.arange(0.1, 15., 0.1)
    momentums = []
    for degree in degrees:
        momentum = b_field * field_length * SPEED_OF_LIGHT / (degree * DEGREE_TO_RADIAN) * 1.e-9 # MeV
        momentums.append(momentum)
    gr = TGraph(len(degrees), np.array(degrees), np.array(momentums))

    min_momentum_gev, max_theta_degree = get_min_momentum()
    tl = TLine(max_theta_degree, 0, max_theta_degree, 15)

    c1 = TCanvas('c1', 'c1', 800, 600)
    gStyle.SetOptStat(0)
    set_margin()
    gPad.SetRightMargin(0.2)
    gPad.SetLogz()
    set_h2_color_style()

    set_h2_style(h1)
    h1.GetYaxis().SetTitle('Momentum (GeV)')
    h1.GetXaxis().SetTitle('Bending Angle (degree)')

    h1.GetXaxis().SetRangeUser(0, 12)
    h1.Draw('colz')

    set_graph_style(gr)
    gr.SetLineColor(kBlue)
    gr.SetLineStyle(7)
    gr.Draw('sames,L')

    tl.SetLineWidth(3)
    tl.SetLineColor(kMagenta + 1)
    tl.SetLineStyle(10)
    tl.Draw()

    lg1 = TLegend(0.17, 0.17, 0.39, 0.29)
    set_legend_style(lg1)
    lg1.AddEntry(gr, 'Calculated P vs. #theta', 'l')
    lg1.AddEntry(tl, 'Calculated maximum #theta', 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_m1_downstream.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_m1_upstream():
    h1 = TH1D('h1', 'h1', 30, 0., 15.)

    tf = TFile('magnet.root')
    for event in tf.Get('VirtualDetector/UpstreamDetector'):
        # momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1.e3      # GeV
        momentum = event.Pz / 1.e3
        h1.Fill(momentum)
        # break

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)
    h1.GetYaxis().SetRangeUser(0, 5000)
    h1.GetXaxis().SetTitle('Momentum (GeV)')
    h1.GetYaxis().SetTitle('Particle Count')
    h1.GetYaxis().SetTitleOffset(1.5)
    h1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_m1_upstream.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_m1_block_momentum():
    h_up = TH1D('h_up', 'h_up', 30, 0., 15.)
    h_downs = []

    filenames = ['magnet.1.8T.root', 'magnet.0.9T.root', 'magnet.0.4T.root']
    colors = [kBlue + 2, kGreen + 2, kRed + 2]
    b_fields = [1.8, 0.9, 0.4]

    for i, filename in enumerate(filenames):
        tf = TFile(filename)
        if i == 0:
            for event in tf.Get('VirtualDetector/UpstreamDetector'):
                momentum = event.Pz / 1.e3
                h_up.Fill(momentum)

        h_down = TH1D('h_down_{}'.format(i), 'h_down_{}'.format(i), 30, 0., 15.)
        for event in tf.Get('VirtualDetector/DownstreamDetector'):
            momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1.e3      # GeV
            h_down.Fill(momentum)
        h_down.SetDirectory(0)
        tf.Close()
        h_downs.append(h_down)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gStyle.SetOptStat(0)
    lg1 = TLegend(0.52, 0.17, 0.75, 0.42)
    set_legend_style(lg1)

    set_h1_style(h_up)
    h_up.GetYaxis().SetRangeUser(0, 4500)
    h_up.GetXaxis().SetTitle('Momentum (GeV)')
    h_up.GetYaxis().SetTitle('Particle Count')
    h_up.GetYaxis().SetTitleOffset(1.5)
    h_up.SetLineColor(kBlack)
    h_up.Draw()
    lg1.AddEntry(h_up, 'Before magnet', 'l')

    for i, h_down in enumerate(h_downs):
        h_down.Draw('sames')
        set_h1_style(h_down)
        h_down.SetLineColor(colors[i])
        lg1.AddEntry(h_down, 'After {} T magnet'.format(b_fields[i]), 'l')

    lg1.Draw()
    c1.Update()
    c1.SaveAs('{}/plot_m1_block_momentum.pdf'.format(FIGURE_DIR))
    c1.SaveAs('{}/plot_m1_block_momentum.png'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_p_vs_angle_max_angle():
    field_length = 42. * INCH_TO_METER # m

    min_momentum_gev, max_theta_degree = get_min_momentum()
    tl = TLine(max_theta_degree, 0, max_theta_degree, 15)

    b_fields = [1.8, 0.9, 0.45]
    colors = [kBlue + 2, kGreen + 2, kRed + 2]
    degrees = np.arange(0.1, 16., 0.1)

    b_field_momentums = []
    for b_field in b_fields:
        momentums = []
        for degree in degrees:
            momentum = b_field * field_length * SPEED_OF_LIGHT / (degree * DEGREE_TO_RADIAN) * 1.e-9 # GeV
            momentums.append(momentum)
        b_field_momentums.append(momentums)

    print(degrees)
    print(momentums)

    grs = []
    for b_field_momentum in b_field_momentums:
        gr = TGraph(len(degrees), np.array(degrees), np.array(b_field_momentum))
        set_graph_style(gr)
        gr.SetMarkerStyle(24)
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    # lg1 = TLegend(0.34, 0.64, 0.58, 0.86)
    lg1 = TLegend(0.6, 0.62, 0.83, 0.84)
    set_legend_style(lg1)
    gPad.SetGrid()

    grs[0].Draw('AL')
    grs[0].GetXaxis().SetRangeUser(0., 16.)
    grs[0].GetYaxis().SetRangeUser(0., 15.)
    grs[0].GetXaxis().SetTitle('Bending Angle #theta (degree)')
    grs[0].GetYaxis().SetTitle('Momentum (GeV)')
    grs[0].GetYaxis().SetTitleOffset(1.5)
    grs[0].SetLineColor(colors[0])
    lg1.AddEntry(grs[0], 'B = {} T'.format(b_fields[0]), 'l')

    for i in range(1, len(grs)):
        grs[i].Draw('sames,L')
        grs[i].SetLineColor(colors[i])
        lg1.AddEntry(grs[i], 'B = {} T'.format(b_fields[i]), 'l')

    tl.SetLineWidth(3)
    tl.SetLineColor(kMagenta + 1)
    tl.SetLineStyle(10)
    # tl.Draw()
    # lg1.AddEntry(tl, '#theta_{max}', 'l')

    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_p_vs_angle_max_angle.pdf'.format(FIGURE_DIR))
    c1.SaveAs('{}/plot_p_vs_angle_max_angle.png'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_max_theta():
    b_field = 1.8               # T
    b_field_length = 42. * INCH_TO_METER # m

    aperture_widths = np.arange(0., 10., 0.1)
    max_theta_degrees = []
    for aperture_width in aperture_widths:
        min_momentum_gev, max_theta_degree = get_min_momentum(b_field=b_field,
                                                              b_field_length=b_field_length,
                                                              aperture_width=aperture_width * INCH_TO_METER)
        max_theta_degrees.append(max_theta_degree)
    gr = TGraph(len(aperture_widths), np.array(aperture_widths), np.array(max_theta_degrees))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()

    set_graph_style(gr)
    gr.GetXaxis().SetTitle('Aperture Width W (inch)')
    gr.GetYaxis().SetTitle('#theta_{max} (degree)')
    gr.Draw('AL')

    c1.Update()
    c1.SaveAs('{}/plot_max_theta.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_min_b_field():
    # aperture_width = 4. * INCH_TO_METER
    b_field_length = 42. * INCH_TO_METER

    aperture_widths = [4., 6.]  # inch
    min_momentum_gevs = np.arange(0.1, 1, 0.1)
    grs = []
    for aperture_width in aperture_widths:
        aperture_width *= INCH_TO_METER
        min_b_fields = []
        for min_momentum_gev in min_momentum_gevs:
            min_momentum_si = min_momentum_gev * 1.e9 / SPEED_OF_LIGHT * ELEMENTARY_CHARGE
            max_theta_radian, max_theta_degree = get_max_theta(b_field_length=b_field_length, aperture_width=aperture_width)
            min_b_field = min_momentum_si * max_theta_radian / ELEMENTARY_CHARGE / b_field_length
            min_b_fields.append(min_b_field)
            # print('min_momentum_si = ', min_momentum_si)
            # print('max_theta_radian = ', max_theta_radian)
            # print('max_theta_degree = ', max_theta_degree)
            # print('min_b_field = ', min_b_field)
        gr = TGraph(len(min_momentum_gevs), np.array(min_momentum_gevs), np.array(min_b_fields))
        set_graph_style(gr)
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    lg1 = TLegend(0.18, 0.74, 0.44, 0.86)
    set_legend_style(lg1)
    gPad.SetGrid()

    grs[0].Draw('AL')
    grs[0].GetXaxis().SetTitle('Minimum Momentum (GeV)')
    grs[0].GetYaxis().SetTitle('Required B Field (T)')
    grs[0].GetYaxis().SetRangeUser(0., 0.8)
    lg1.AddEntry(grs[0], '{:.0f} inch wide aperture'.format(aperture_widths[0]), 'l')

    grs[1].Draw('sames,L')
    grs[1].SetLineColor(kBlue + 1)
    lg1.AddEntry(grs[1], '{:.0f} inch wide aperture'.format(aperture_widths[1]), 'l')

    lg1.Draw()
    c1.Update()
    # c1.SaveAs('{}/plot_min_b_field.pdf'.format(FIGURE_DIR))
    c1.SaveAs('{}/plot_min_b_field.png'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_pdg_pxy_thetas(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))

    pdgs = [11, -11, 13, -13, 211, -211, 321, -321, 2212, -2212, 2112, -2112, 22]
    h_pdg_pxy_thetas = {}
    h_pdg_pxys = {}

    for pdg in pdgs:
        name = PDG.GetParticle(pdg).GetName()
        h_pxy_theta = TH2D('h_pxy_theta_{}'.format(name), 'h_pxy_theta_{}'.format(name), 90, -90, 90, 60, 0, 3)
        set_h2_style(h_pxy_theta)
        h_pxy_theta.SetDirectory(0)
        h_pdg_pxy_thetas[pdg] = h_pxy_theta

        h_pxy = TH1D('h_pxy_{}'.format(name), 'h_pxy_{}'.format(name), 100, 0, 2)
        set_h1_style(h_pxy)
        h_pxy.SetDirectory(0)
        h_pdg_pxys[pdg] = h_pxy

    event_count = 0
    for event in tf.Get('Detector/Detector'):
        theta = atan(event.Px / event.Py)
        theta_degree = theta * 180. / pi
        momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1.e3      # GeV
        if event.PDGid in pdgs:
            h_pdg_pxy_thetas[event.PDGid].Fill(theta_degree, momentum)

        event_count += 1
        if event_count % 10000 == 0:
            print('event_count = {}'.format(event_count))

    return h_pdg_pxy_thetas


def plot_pxy_theta(h_pdg_pxy_thetas, pdg, filename):
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h2_color_style()

    gStyle.SetOptStat(0)
    gPad.SetRightMargin(0.2)
    h_pdg_pxy_thetas[pdg].Draw('colz')

    c1.Update()
    c1.SaveAs('{}/plot_pxy_theta.{}.{}.pdf'.format(FIGURE_DIR, filename, PDG.GetParticle(pdg).GetName()))
    # input('Press any key to continue.')


def plot_pxy_thetas(filename):
    h_pdg_pxy_thetas = get_pdg_pxy_thetas(filename)

    pdgs = [11, -11, -13, 13, 211, -211, 2212, -2212, 2112, -2112, 22]
    for pdg in pdgs:
        plot_pxy_theta(h_pdg_pxy_thetas, pdg, filename)


def plot_momentum_pxy_theta(h_momentum_pdg_pxy_thetas, pdg):
    c1 = TCanvas('c1', 'c1', 1000, 800)
    gStyle.SetOptStat(0)
    set_margin()
    set_h2_color_style()

    hs = []
    texs = []
    for momentum in h_momentum_pdg_pxy_thetas.keys():
        hs.append(h_momentum_pdg_pxy_thetas[momentum][pdg])
        tex = TLatex(85, 2.9, '{} GeV'.format(momentum))
        tex.SetTextFont(43)
        tex.SetTextSize(25)
        tex.SetTextAlign(33)
        texs.append(tex)

    c1.Divide(2, 2)
    c1.cd(1)
    gPad.SetRightMargin(0.2)
    gPad.SetTopMargin(0.05)
    gPad.SetBottomMargin(0.15)
    # gPad.SetLogz()

    hs[0].Draw('colz')
    hs[0].GetYaxis().SetTitle('Momentum (GeV)')
    hs[0].GetYaxis().SetTitleOffset(1.25)
    texs[0].Draw()

    c1.cd(2)
    gPad.SetRightMargin(0.2)
    gPad.SetTopMargin(0.05)
    gPad.SetBottomMargin(0.15)
    hs[1].Draw('colz')
    texs[1].Draw()

    c1.cd(3)
    gPad.SetRightMargin(0.2)
    gPad.SetTopMargin(0.05)
    gPad.SetBottomMargin(0.15)
    hs[2].Draw('colz')
    hs[2].GetYaxis().SetTitle('Momentum (GeV)')
    hs[2].GetXaxis().SetTitle('Angle (degree)')
    hs[2].GetXaxis().SetTitleOffset(2.05)
    hs[2].GetYaxis().SetTitleOffset(1.25)
    texs[2].Draw()

    c1.cd(4)
    gPad.SetRightMargin(0.2)
    gPad.SetTopMargin(0.05)
    gPad.SetBottomMargin(0.15)
    hs[3].Draw('colz')
    hs[3].GetXaxis().SetTitle('Angle (degree)')
    hs[3].GetXaxis().SetTitleOffset(2.05)
    texs[3].Draw()

    c1.Update()
    c1.SaveAs('{}/plot_momentum_pxy_theta.{}.pdf'.format(FIGURE_DIR, PDG.GetParticle(pdg).GetName()))
    # input('Press any key to continue.')


def plot_momentum_pxy_thetas():
    h_momentum_pdg_pxy_thetas = {}
    momentums = [8, 16, 32, 64]
    for momentum in momentums:
        h_momentum_pdg_pxy_thetas[momentum] = get_pdg_pxy_thetas('target.{}GeV.root'.format(momentum))

    pdgs = [11, -11, 13, -13, 211, -211, 321, -321, 2212, -2212, 2112, -2112, 22]
    # pdgs = [22]
    for pdg in pdgs:
        plot_momentum_pxy_theta(h_momentum_pdg_pxy_thetas, pdg)


def print_slide_momentum_pxy_thetas():
    pdgs = [11, -11, 13, -13, 211, -211, 321, -321, 2212, -2212]
    with open('{}/../momentum_pxy_thetas.tex'.format(FIGURE_DIR), 'w') as f_momentum:
        for pdg in pdgs:
            particle_name = PDG.GetParticle(pdg).GetName()
            f_momentum.write('\\begin{frame}\n')
            f_momentum.write('  \\frametitle{{P vs. Angle at Various Beam Energies for {}}}\n'.format(particle_name))
            f_momentum.write('  \\begin{figure}\n')
            f_momentum.write('    \\includegraphics[width=10.5cm]{{{{figures/plot_momentum_pxy_theta.{}}}.pdf}}\n'.format(particle_name))
            f_momentum.write('  \\end{figure}\n')
            f_momentum.write('\\end{frame}\n')
            f_momentum.write('\n% .........................................................\n\n')


def save_particle_to_csv(filename):
    tf1 = TFile('{}/{}'.format(DATA_DIR, filename))

    pid_momentums = {}
    particles = []
    noise_particles = []

    keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
    for key in keys:
        print('key = {}'.format(key))
        track_count = 0
        for track in tf1.Get(key):
            track_count += 1
            pass_all = track.TrackPresentstart_line and \
                       track.TrackPresenttof_us and \
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

    with open('{}/{}.csv'.format(DATA_DIR, filename), 'w') as f_fraction:
        for particle in particles:
            f_fraction.write('0,{}\n'.format(','.join(list(map(str, particle)))))
        for noise_particle in noise_particles:
            f_fraction.write('1,{}\n'.format(','.join(list(map(str, noise_particle)))))

    pprint(pid_momentums)


def plot_particle_momentum(filename, x_min, x_max, **kwargs):
    bin_count = kwargs.get('bin_count', 50)
    y_max = kwargs.get('y_max', 0.)
    y_title = kwargs.get('y_title', 'Particle Count')
    log_y = kwargs.get('log_y', False)
    plot_noise = kwargs.get('plot_noise', False)
    normalization_factor = kwargs.get('normalization_factor', 1.)
    y_title_offset = kwargs.get('y_title_offset', 1.8)
    title = kwargs.get('title', '64 GeV Secondary Beam')

    count_precision = 0 if normalization_factor == 1. else 1

    pid_momentums = {}
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            # row.insert(0, 0)

            is_noise = int(row[0])
            if plot_noise and not is_noise:
                continue
            if not plot_noise and is_noise:
                continue

            pid = int(float(row[-2]))
            px = float(row[-5])
            py = float(row[-4])
            pz = float(row[-3])
            momentum = (px**2 + py**2 + pz**2)**0.5

            if pid not in pid_momentums:
                pid_momentums[pid] = [momentum]
            else:
                pid_momentums[pid].append(momentum)

    pid_counts = []
    pid_hists = {}
    h_all = TH1D('h_all', 'h_all', bin_count, x_min, x_max)
    for pid, momentums in pid_momentums.items():
        if pid == 22:
            continue

        pid_counts.append([pid, len(momentums)])
        hist = TH1D('h_{}'.format(pid), 'h_{}'.format(pid), 50, x_min, x_max)
        for momentum in momentums:
            hist.Fill(momentum)
            h_all.Fill(momentum)

        hist.Scale(1. / normalization_factor)
        pid_hists[pid] = hist
    h_all.Scale(1. / normalization_factor)

    if y_max == 0.:
        for pid, hist in pid_hists.items():
            if hist.GetMaximum() > y_max:
                y_max = hist.GetMaximum()
    y_max *= 1.2

    # c1 = TCanvas('c1', 'c1', 800, 800)
    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gStyle.SetOptStat(0)
    if log_y:
        gPad.SetLogy()

    colors = [
        kBlack,
        kRed,
        kBlue,
        kMagenta + 2,
        kViolet + 2,
        kAzure + 2,
        kCyan + 2,
        kTeal + 2,
        kGreen + 2,
        kSpring + 2,
        kYellow + 2,
        kOrange + 2
    ]

    lg1 = None
    if plot_noise:
        lg1 = TLegend(0.4, 0.5, 0.7, 0.87)
    else:
        # lg1 = TLegend(0.575, 0.6, 0.84, 0.84)
        # lg1 = TLegend(0.545, 0.6, 0.84, 0.84)
        lg1 = TLegend(0.51, 0.6, 0.84, 0.84)
    set_legend_style(lg1)

    pid_counts = sorted(pid_counts, key=lambda x: x[1], reverse=True)
    pids = [pid_count[0] for pid_count in pid_counts]
    for i, pid in enumerate(pids):
        hist = pid_hists[pid]
        set_h1_style(hist)
        hist.SetLineColor(colors[i])

        if plot_noise:
            lg1.AddEntry(hist, '{} ({:.0f}, {:.0f} MeV)'.format(PDG.GetParticle(pid).GetName(), hist.Integral(), hist.GetMean()), 'l')
        else:
            lg1.AddEntry(hist, '{1} ({2:.{0}f})'.format(count_precision, PDG.GetParticle(pid).GetName(), hist.Integral()), 'l')

        if i == 0:
            hist.Draw('hist')
            hist.GetXaxis().SetTitle('Momentum (MeV)')
            hist.GetYaxis().SetTitle(y_title)
            hist.GetYaxis().SetTitleOffset(y_title_offset)
            hist.SetTitle(title)
            if y_max:
                hist.GetYaxis().SetRangeUser(0 if not log_y else 0.5, y_max)
        else:
            hist.Draw('hist,sames')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(43)
    latex.SetTextSize(28)
    # latex.DrawLatex(0.2, 0.83, label)

    x_latex = 0.53
    if not plot_noise:
        latex.DrawLatex(x_latex, 0.42, 'rms = {:.0f} MeV'.format(h_all.GetRMS()))
        latex.DrawLatex(x_latex, 0.48, 'mean = {:.0f} MeV'.format(h_all.GetMean()))
        latex.DrawLatex(x_latex, 0.54, 'total count = {1:.{0}f}'.format(count_precision, h_all.Integral()))
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_particle_momentum.{}.plot_noise_{}.pdf'.format(FIGURE_DIR, filename, plot_noise))
    input('Press any key to continue.')


def get_kalman_data(velocity, z_count, z_var):
    np.random.seed(seed=1)
    random_dzs = np.random.normal(0., z_var, z_count)

    ts = []
    zs = []
    random_zs = []
    for i in range(1, z_count + 1):
        t = float(i)
        z = velocity * t
        ts.append(t)
        zs.append(z)

        z += random_dzs[i - 1]
        random_zs.append(z)
    return ts, zs, random_zs


def test_1d_kalman():
    velocity = 1.
    dt = 1.
    pos = 1.
    pos_var = 10.
    process_var = 1.
    z_var = 1.
    z_count = 10

    ts, zs, random_zs = get_kalman_data(velocity, z_count, z_var)
    z_count = len(zs)
    gr_true = TGraph(z_count, np.array(ts), np.array(zs))
    gr_data = TGraph(z_count, np.array(ts), np.array(random_zs))

    predicts = []
    filters = []
    filter_pluss = []
    filter_minuss = []

    for i in range(z_count):
        dx = velocity * dt
        if i > 0:
            pos += velocity * dt
            pos_var += process_var
        predicts.append(pos)

        pos = (pos_var * random_zs[i] + z_var * pos) / (pos_var + z_var)
        pos_var = pos_var * z_var / (pos_var + z_var)
        filters.append(pos)
        filter_pluss.append(pos + pos_var)
        filter_minuss.append(pos - pos_var)

    gr_predict = TGraph(z_count, np.array(ts), np.array(predicts))
    gr_filter = TGraph(z_count, np.array(ts), np.array(filters))
    gr_filter_var = TGraph(2 * z_count)
    for i in range(z_count):
        gr_filter_var.SetPoint(i, ts[i], filter_pluss[i])
        gr_filter_var.SetPoint(z_count + i, ts[z_count - i - 1], filter_minuss[z_count -i - 1])

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    set_graph_style(gr_true)
    gr_true.GetYaxis().SetRangeUser(0, len(ts) * 1.2)
    gr_true.GetYaxis().SetNdivisions(505, 1)
    gr_true.GetXaxis().SetNdivisions(505, 1)
    gr_true.SetLineStyle(7)
    gr_true.SetLineColor(kBlue)
    # gr_true.Draw('AL')

    set_graph_style(gr_data)
    gr_data.SetMarkerStyle(20)
    gr_data.SetMarkerSize(1)
    gr_data.SetMarkerColor(kBlack)
    gr_data.GetYaxis().SetRangeUser(0, len(ts) * 1.1)
    gr_data.GetXaxis().SetTitle('Time (s)')
    gr_data.GetYaxis().SetTitle('Position (m)')
    gr_data.Draw('AP')

    set_graph_style(gr_predict)
    gr_predict.SetMarkerStyle(21)
    gr_predict.SetMarkerSize(1)
    gr_predict.SetMarkerColor(kRed + 1)
    gr_predict.Draw('P')

    set_graph_style(gr_filter)
    gr_filter.SetMarkerStyle(22)
    gr_filter.SetMarkerSize(1)
    gr_filter.SetMarkerColor(kBlue)
    gr_filter.SetLineColor(kBlue)
    gr_filter.Draw('L')

    gr_filter_var.SetFillStyle(3001)
    gr_filter_var.SetFillColor(16)
    gr_filter_var.SetLineWidth(0)
    gr_filter_var.Draw('F')
    gr_data.Draw('P')
    gr_predict.Draw('P')

    lg1 = TLegend(0.18, 0.67, 0.46, 0.88)
    set_legend_style(lg1)
    lg1.AddEntry(gr_data, 'data', 'p')
    lg1.AddEntry(gr_predict, 'prediction', 'p')
    lg1.AddEntry(gr_filter, 'filter', 'l')
    lg1.AddEntry(gr_filter_var, 'variance', 'f')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/test_1d_kalman.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def test_1d_kalman_prediction_only():
    velocity = 1.
    dt = 1.
    pos = 1.
    pos_var = 1.
    process_var = 1.
    z_var = 1.
    z_count = 10

    ts, zs, random_zs = get_kalman_data(velocity, z_count, z_var)
    z_count = len(zs)

    filters = []
    filter_pluss = []
    filter_minuss = []

    for i in range(z_count):
        dx = velocity * dt
        if i > 0:
            pos += velocity * dt
            pos_var += process_var
        filters.append(pos)
        filter_pluss.append(pos + pos_var)
        filter_minuss.append(pos - pos_var)

    gr_filter = TGraph(z_count, np.array(ts), np.array(filters))
    gr_filter_var = TGraph(2 * z_count)
    for i in range(z_count):
        gr_filter_var.SetPoint(i, ts[i], filter_pluss[i])
        gr_filter_var.SetPoint(z_count + i, ts[z_count - i - 1], filter_minuss[z_count -i - 1])

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    set_graph_style(gr_filter)
    gr_filter.GetYaxis().SetRangeUser(-2, len(ts) * 1.5)
    gr_filter.GetYaxis().SetNdivisions(505, 1)
    gr_filter.GetXaxis().SetNdivisions(510, 1)
    gr_filter.GetXaxis().SetTitle('Time (s)')
    gr_filter.GetYaxis().SetTitle('Position (m)')
    gr_filter.SetLineColor(kBlue)
    gr_filter.Draw('AL')

    gr_filter_var.SetFillStyle(3001)
    gr_filter_var.SetFillColor(16)
    gr_filter_var.SetLineWidth(0)
    gr_filter_var.Draw('F')

    lg1 = TLegend(0.18, 0.67, 0.46, 0.88)
    set_legend_style(lg1)
    lg1.AddEntry(gr_filter, 'prediction alone', 'l')
    lg1.AddEntry(gr_filter_var, 'variance', 'f')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/test_1d_kalman_prediction_alone.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def test_graph_shade():
    xs = []
    y_1s = []
    y_2s = []
    for i in range(10):
        xs.append(float(i))
        y_1s.append(1.)
        y_2s.append(2.)
    gr1 = TGraph(len(xs), np.array(xs), np.array(y_1s))
    gr2 = TGraph(len(xs), np.array(xs), np.array(y_2s))
    gr = get_graph_shade(gr1, gr2)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    gr.SetFillStyle(3001)
    gr.SetFillColor(16)
    gr.SetLineWidth(0)
    gr.Draw('AF')

    c1.Update()
    c1.SaveAs('figures/test_gr_shade.pdf')
    input('Press any key to continue.')


def plot_cherenkov_index_of_refaction():
    # names = ['proton', 'K+', 'pi+', 'mu+', 'e+']
    # colors = [kRed + 2, kMagenta + 2, kBlue + 2, kGreen + 2, kBlack]
    names = ['pi+', 'mu+', 'e+']
    colors = [kBlue + 2, kGreen + 2, kBlack]
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    eta = 4.1e-4                  # atm-1

    momentums = np.arange(0.01, 10, 0.01)
    ppressures = []
    for i, mass in enumerate(masses):
        pressures = []
        for momentum in momentums:
            # pressure = 1. / eta * ((1 + (mass / momentum)**2)**0.5 - 1.)
            pressure = (1 + (mass / momentum)**2)**0.5
            pressures.append(pressure)
        ppressures.append(pressures)

    grs = []
    for i in range(len(ppressures)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(ppressures[i]))
        set_graph_style(gr)
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    # gPad.SetLogy()
    gPad.SetGrid()
    gPad.SetLeftMargin(0.2)

    # lg1 = TLegend(0.2, 0.8, 0.88, 0.88)
    lg1 = TLegend(0.3, 0.18, 0.8, 0.26)
    set_legend_style(lg1)
    lg1.SetNColumns(5)

    grs[0].Draw('AL')
    grs[0].SetLineColor(colors[0])
    grs[0].GetXaxis().SetRangeUser(0., 3)
    grs[0].GetYaxis().SetRangeUser(1., 1.004)
    # grs[0].GetYaxis().SetRangeUser(0.5, 10)
    # grs[0].GetYaxis().SetRangeUser(1.e-5, 1e6)
    grs[0].GetYaxis().SetDecimals()
    grs[0].GetYaxis().SetTitleOffset(2)

    grs[0].GetYaxis().SetTitle('Index of Refraction')
    grs[0].GetXaxis().SetTitle('Momentum (GeV)')
    lg1.AddEntry(grs[0], names[0], 'l')
    for i in range(1, len(names)):
        grs[i].Draw('sames,L')
        grs[i].SetLineColor(colors[i])
        lg1.AddEntry(grs[i], names[i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_cherenkov_index_of_refaction.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_cherenkov_photon_count():
    index_of_refraction = 1.0004
    beta = 1.
    theta = math.acos(1. / index_of_refraction / beta) * 180. / pi
    sin_square_theta = 1. - (1. / index_of_refraction / beta)**2.

    pmt_quantum_efficiency = 0.2
    dndx = 2. * pi * 1. / 137. * sin_square_theta * pmt_quantum_efficiency * (1. / 300 - 1. / 500.) * 1.e9
    length = 2.
    efficiency = 0.8
    dn = dndx * length * efficiency

    # radius = 0.1
    # area = 2. * pi * radius * length
    # n_per_area = dn / area
    # pmt_area = 0.1**2
    # n_pmt = pmt_area * n_per_area

    print('theta = {} degree'.format(theta))
    print('sin_square_theta = {}'.format(sin_square_theta))
    print('dndx = {}'.format(dndx))
    print('dn = {}'.format(dn))
    # print('area = {}'.format(area))
    # print('n_per_area= {}'.format(n_per_area))
    # print('n_pmt = {}'.format(n_pmt))


def plot_time_of_flight_mc(**kwargs):
    distance = kwargs.get('distance', 12.8) # m
    y_min = kwargs.get('y_min', 9.9e3)
    y_max = kwargs.get('y_max', 2.e5)

    # simulation
    filenames = [
        'beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.45T.10m.root.csv',
        'beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.9T.10m.root.csv',
        'beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.35T.10m.root.csv',
        'beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.8T.10m.root.csv'
    ]
    pid_tof_momentums = {}
    h_tof_momentum = TH2D('h_tof_momentum', 'h_tof_momentum', 170, 30, 200, 300, 0, 3)
    # h_tof_momentum.Rebin2D(2, 2)

    for filename in filenames:
        with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
            for row in csv.reader(f_csv, delimiter=','):
                pid = int(float(row[-2]))
                tof = (float(row[7]) - float(row[2])) * 1.e9
                px = float(row[-5])
                py = float(row[-4])
                pz = float(row[-3])
                momentum = (px**2 + py**2 + pz**2)**0.5 / 1.e3
                tof_momentum = (tof, momentum)

                h_tof_momentum.Fill(tof, momentum)
                if pid not in pid_tof_momentums:
                    pid_tof_momentums[pid] = [tof_momentum]
                else:
                    pid_tof_momentums[pid].append(tof_momentum)

    # pid_graphs = {}
    # for pid, tof_momentums in pid_tof_momentums.items():
    #     tofs = [tof_momentum[0] for tof_momentum in tof_momentums]
    #     momentums = [tof_momentum[1] for tof_momentum in tof_momentums]
    #     gr = TGraph(len(tofs), np.array(tofs), np.array(momentums))
    #     pid_graphs[pid] = gr

    # calculation
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
        gr = TGraph(len(momentums), np.array(ttofs[i]), np.array(momentums))
        gr.SetLineColor(colors[i])
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 800, 600)
    gStyle.SetOptStat(0)
    set_margin()
    gPad.SetGrid()

    set_h2_color_style()
    set_h2_style(h_tof_momentum)
    h_tof_momentum.Draw('colz')
    h_tof_momentum.GetYaxis().SetTitle('Momentum (GeV)')
    h_tof_momentum.GetXaxis().SetTitle('Time of Flight (ns)')

    lg1 = TLegend(0.65, 0.5, 0.9, 0.85)
    set_legend_style(lg1)

    for i in range(0, len(names)):
        set_graph_style(grs[i])
        grs[i].SetLineStyle(7)
        grs[i].SetLineWidth(1)
        grs[i].Draw('sames,L')
        lg1.AddEntry(grs[i], names[i], 'l')
    lg1.Draw()
    h_tof_momentum.Draw('colz,sames')

    c1.Update()
    c1.SaveAs('{}/plot_time_of_flight_mc.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_particle_angle(filename):
    pid_momentums = {}
    h_angle = TH2D('h_angle', 'h_angle', 100, -2, 2, 100, -2, 2)
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            px = float(row[-5])
            py = float(row[-4])
            pz = float(row[-3])
            angle_x = atan(px / pz) * RADIAN_TO_DEGREE
            angle_y = atan(py / pz) * RADIAN_TO_DEGREE
            h_angle.Fill(angle_x, angle_y)

    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    gPad.SetRightMargin(0.15)
    gStyle.SetOptStat(0)

    set_h2_color_style()
    set_h2_style(h_angle)
    h_angle.Draw('colz')
    h_angle.GetXaxis().SetTitle('Horizontal Angle (P_{x} / P_{z}, degree)')
    h_angle.GetYaxis().SetTitle('Vertical Angle (P_{y} / P_{z}, degree)')
    h_angle.GetXaxis().SetTitleOffset(1.4)
    h_angle.GetYaxis().SetTitleOffset(1.4)

    c1.Update()
    c1.SaveAs('{}/plot_particle_angle.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_noise_particle(filename, **kwargs):
    log_y = kwargs.get('log_y', False)
    show_boundary = kwargs.get('show_boundary', False)

    width = 2606.2 / 10.       # cm
    half_width = width / 2.    # cm
    x0 = -1354.4 / 10.         # cm
    y0 = 0.
    margin = 20.
    pid_y_x_hists = {}
    pid_momentum_x_hists = {}

    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            is_noise = int(row[0])
            if not is_noise:
                continue

            pid = int(float(row[-2]))
            px = float(row[-5])
            py = float(row[-4])
            pz = float(row[-3])
            momentum = (px**2 + py**2 + pz**2)**0.5
            x = float(row[-9])
            y = float(row[-8])
            z = float(row[-7])

            if pid not in pid_y_x_hists:
                pid_y_x_hists[pid] = TH2D('h_y_x_{}'.format(pid), 'h_y_x_{}'.format(pid), 100, x0 - half_width - margin, x0 + half_width + margin, 100, y0 - half_width - margin, y0 + half_width + margin)
            if pid not in pid_momentum_x_hists:
                pid_momentum_x_hists[pid] = TH2D('h_momentum_x_{}'.format(pid), 'h_momentum_x_{}'.format(pid), 100, x0 - half_width - margin, x0 + half_width + margin, 100, 0, 3000)

            pid_y_x_hists[pid].Fill(x / 10., y / 10.)
            pid_momentum_x_hists[pid].Fill(x / 10., momentum)

    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    set_h2_color_style()
    gPad.SetRightMargin(0.15)

    for pid, h1 in pid_y_x_hists.items():
        set_h2_style(h1)
        h1.Draw('colz')

        if show_boundary:
            tl_left = TLine(x0 - half_width, y0 - half_width, x0 - half_width, y0 + half_width)
            tl_right = TLine(x0 + half_width, y0 - half_width, x0 + half_width, y0 + half_width)
            tl_top = TLine(x0 - half_width, y0 + half_width, x0 + half_width, y0 + half_width)
            tl_bottom = TLine(x0 - half_width, y0 - half_width, x0 + half_width, y0 - half_width)
            tls = [tl_left, tl_right, tl_top, tl_bottom]
            for tl in tls:
                tl.SetLineColor(kRed)
                tl.SetLineWidth(3)
                tl.Draw()

        h1.GetXaxis().SetTitle('X (cm)')
        h1.GetYaxis().SetTitle('Y (cm)')
        h1.GetXaxis().SetTitleOffset(1.8)
        h1.GetYaxis().SetTitleOffset(2.)
        c1.Update()
        c1.SaveAs('{}/plot_noise_particle_y_x.{}.pid_{}.pdf'.format(FIGURE_DIR, filename, PDG.GetParticle(pid).GetName()))

    for pid, h1 in pid_momentum_x_hists.items():
        set_h2_style(h1)
        h1.Draw('colz')
        h1.GetXaxis().SetTitle('X (cm)')
        h1.GetYaxis().SetTitle('Momentum (MeV)')
        h1.GetXaxis().SetTitleOffset(1.8)
        h1.GetYaxis().SetTitleOffset(2.)
        c1.Update()
        c1.SaveAs('{}/plot_noise_particle_momentum_x.{}.pid_{}.pdf'.format(FIGURE_DIR, filename, PDG.GetParticle(pid).GetName()))
    input('Press any key to continue.')


def plot_trigger_particle(filename, **kwargs):
    show_boundary = kwargs.get('show_boundary', False)

    width = 2606.2 / 10.       # cm
    half_width = width / 2.    # cm
    x0 = -1354.4 / 10.         # cm
    y0 = 0.
    margin = 20.
    h1 = TH2D('h1', 'h1', 200, x0 - half_width - margin, x0 + half_width + margin, 200, y0 - half_width - margin, y0 + half_width + margin)
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            is_noise = int(row[0])
            if is_noise:
                continue

            pid = int(float(row[-2]))
            px = float(row[-5])
            py = float(row[-4])
            pz = float(row[-3])
            momentum = (px**2 + py**2 + pz**2)**0.5
            x = float(row[-9])
            y = float(row[-8])
            z = float(row[-7])

            h1.Fill(x / 10., y / 10.)

    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    set_h2_color_style()
    gPad.SetRightMargin(0.15)

    set_h2_style(h1)
    h1.Draw('colz')
    if show_boundary:
        tl_left = TLine(x0 - half_width, y0 - half_width, x0 - half_width, y0 + half_width)
        tl_right = TLine(x0 + half_width, y0 - half_width, x0 + half_width, y0 + half_width)
        tl_top = TLine(x0 - half_width, y0 + half_width, x0 + half_width, y0 + half_width)
        tl_bottom = TLine(x0 - half_width, y0 - half_width, x0 + half_width, y0 - half_width)
        tls = [tl_left, tl_right, tl_top, tl_bottom]
        for tl in tls:
            tl.SetLineColor(kRed)
            tl.SetLineWidth(3)
            tl.Draw()
    else:
        h1.GetXaxis().SetRangeUser(-215, -65)
        h1.GetYaxis().SetRangeUser(-75, 75)

    h1.GetXaxis().SetTitle('X (cm)')
    h1.GetYaxis().SetTitle('Y (cm)')
    h1.GetXaxis().SetTitleOffset(1.8)
    h1.GetYaxis().SetTitleOffset(2.)
    c1.Update()
    c1.SaveAs('{}/plot_trigger_particle.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def compare_particle_count():
    protons = [26479, 2223, 2317, 2239, 1992, 187, 74, 45, 35]
    pis = [16287, 1328, 1287, 1263, 1123, 100, 51, 24, 22]
    for i in range(len(protons)):
        print('(protons[i] - pis[i]) / pis[i] = {:.1f}'.format((protons[i] - pis[i]) / pis[i] * 100.))
    print('sum(protons) = {}'.format(sum(protons)))
    print('sum(pis) = {}'.format(sum(pis)))
    print('(sum(protons) - sum(pis)) / sum(pis) = {}'.format((sum(protons) - sum(pis)) / sum(pis)))


def get_pid_counts(filename, **kwargs):
    # normalization_factor = kwargs.get('normalization_factor', 1.)

    pid_counts = {}
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            is_noise = int(row[0])
            if is_noise:
                continue

            pid = int(float(row[-2]))
            # px = float(row[-5])
            # py = float(row[-4])
            # pz = float(row[-3])
            # momentum = (px**2 + py**2 + pz**2)**0.5
            # x = float(row[-9])
            # y = float(row[-8])
            # z = float(row[-7])

            if pid not in pid_counts:
                pid_counts[pid] = 1
            else:
                pid_counts[pid] += 1

    # for pid in pid_counts:
    #     pid_counts[pid] /= normalization_factor

    return pid_counts


def get_particle_count_vs_secondary_beam_energy(**kwargs):
    csv_64gev = kwargs.get('csv_64gev', 'beamline.py.in.job_1_1800.18m.b_-0.9T.pi+_64gev.root.csv')
    norm_64gev = kwargs.get('norm_64gev', 1.8)
    csv_32gev = kwargs.get('csv_32gev', 'beamline.py.in.job_1_1800.27m.b_-0.9T.pi+_32gev.root.csv')
    norm_32gev = kwargs.get('norm_32gev', 2.7)
    csv_16gev = kwargs.get('csv_16gev', 'beamline.py.in.job_1_900.45m.b_-0.9T.pi+_16gev.root.csv')
    norm_16gev = kwargs.get('norm_16gev', 4.5)
    csv_8gev = kwargs.get('csv_8gev', 'beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root.csv')
    norm_8gev = kwargs.get('norm_8gev', 9)
    suffix = kwargs.get('suffix', 'b_-0.9T')

    pid_count_64gev = get_pid_counts(csv_64gev)
    pid_count_32gev = get_pid_counts(csv_32gev)
    pid_count_16gev = get_pid_counts(csv_16gev)
    pid_count_8gev = get_pid_counts(csv_8gev)

    pid_counts = [pid_count_8gev, pid_count_16gev, pid_count_32gev, pid_count_64gev]
    beam_energies = [8., 16., 32., 64.]
    norms = [norm_8gev, norm_16gev, norm_32gev, norm_64gev]
    beam_energy_errs = [0., 0., 0., 0.]

    total_counts = []
    pi_counts = []
    proton_counts = []

    total_count_errs = []
    pi_count_errs = []
    proton_count_errs = []
    for i, pid_count in enumerate(pid_counts):
        norm = norms[i]

        total_count = sum(pid_count.values())
        total_counts.append(total_count / norm)
        total_count_errs.append(sqrt(total_count) / norm)

        pi_count = pid_count[211]
        pi_counts.append(pi_count / norm)
        pi_count_errs.append(sqrt(pi_count) / norm)

        proton_count = pid_count[2212]
        proton_counts.append(proton_count / norm)
        proton_count_errs.append(sqrt(proton_count) / norm)

    gr_total = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(total_counts), np.array(beam_energy_errs), np.array(total_count_errs))
    gr_pi = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(pi_counts), np.array(beam_energy_errs), np.array(pi_count_errs))
    gr_proton = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(proton_counts), np.array(beam_energy_errs), np.array(proton_count_errs))

    print('proton_counts = {}'.format(proton_counts))
    print('pi_counts = {}'.format(pi_counts))
    print('total_counts = {}'.format(total_counts))

    grs = [gr_total, gr_pi, gr_proton]
    for gr in grs:
        set_graph_style(gr)
        gr.GetXaxis().SetTitle('Secondary Beam Energy (GeV)')
        gr.GetYaxis().SetTitle('Particle Count per 1M Secondary Beam Particles')
        gr.GetYaxis().SetTitleOffset(2.)

    filename = 'get_particle_count_vs_secondary_beam_energy.{}.root'.format(suffix)
    f_out = TFile(filename, 'RECREATE')
    gr_total.Write('gr_total')
    gr_pi.Write('gr_pi')
    gr_proton.Write('gr_proton')
    f_out.Close()
    print('Graphs saved to {}.'.format(filename))

    return gr_total, gr_pi, gr_proton


def plot_particle_count_vs_secondary_beam_energy(gr_name, **kwargs):
    y_min = kwargs.get('y_min', 0)
    y_max = kwargs.get('y_max', 30)

    filenames = [
        'get_particle_count_vs_secondary_beam_energy.b_-1.8T.root',
        'get_particle_count_vs_secondary_beam_energy.b_-1.35T.root',
        'get_particle_count_vs_secondary_beam_energy.b_-0.9T.root',
        'get_particle_count_vs_secondary_beam_energy.b_-0.45T.root'
    ]

    gr_totals = []
    for filename in filenames:
        tf = TFile(filename)
        gr_totals.append(tf.Get(gr_name))

    lg_names = ['-1.8', '-1.35', '-0.9', '-0.45']
    colors = [kBlack, kRed, kBlue, kGreen + 2]

    lg = TLegend(0.2, 0.65, 0.5, 0.87)
    set_legend_style(lg)

    canvas_width = 800
    canvas_height = 800
    # if gr_name == 'gr_total':
        # canvas_width = 1100
    c1 = TCanvas('c1', 'c1', canvas_width, canvas_height)

    set_margin()
    for i, gr_total in enumerate(gr_totals):
        gr_total.SetLineColor(colors[i])
        gr_total.SetMarkerColor(colors[i])
        lg.AddEntry(gr_total, 'B = {} T'.format(lg_names[i]), 'l')

        if i == 0:
            gr_total.Draw('ALP')
            gr_total.GetYaxis().SetRangeUser(y_min, y_max)
        gr_total.Draw('sames,LP')

    lg.Draw()
    c1.Update()
    c1.SaveAs('{}/plot_particle_count_vs_secondary_beam_energy.{}.pdf'.format(FIGURE_DIR, gr_name))
    input('Press any key to continue.')


def plot_radiation_position(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    pid_hs = {
        'wall': {},
        'cap_start': {},
        'cap_end': {}
    }

    for event in tf.Get('VirtualDetector/wall'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        th2d_name = 'h_wall_{}'.format(pdg_name)

        theta = atan(event.y / event.x)
        if event.x >= 0.:
            theta = pi / 2. - theta
        if event.x < 0:
            theta = -(pi / 2. + theta)
        theta = theta * 180. / pi
        z = event.z / 1000

        if pdg_id not in pid_hs['wall']:
            pid_hs['wall'][pdg_id] = TH2D(th2d_name, th2d_name, 100, -1, 20, 100, -180, 180)
            set_h2_style(pid_hs['wall'][pdg_id])
            pid_hs['wall'][pdg_id].Fill(z, theta)
        else:
            pid_hs['wall'][pdg_id].Fill(z, theta)

    for event in tf.Get('VirtualDetector/cap_start'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        th2d_name = 'h_cap_start_{}'.format(pdg_name)
        y = event.y / 1000.
        x = event.x / 1000.
        if pdg_id not in pid_hs['cap_start']:
            pid_hs['cap_start'][pdg_id] = TH2D(th2d_name, th2d_name, 100, -3, 3, 100, -3, 3)
            set_h2_style(pid_hs['cap_start'][pdg_id])
            pid_hs['cap_start'][pdg_id].Fill(y, x)
        else:
            pid_hs['cap_start'][pdg_id].Fill(y, x)

    for event in tf.Get('VirtualDetector/cap_end'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        th2d_name = 'h_cap_end_{}'.format(pdg_name)
        y = event.y / 1000.
        x = event.x / 1000.
        if pdg_id not in pid_hs['cap_end']:
            pid_hs['cap_end'][pdg_id] = TH2D(th2d_name, th2d_name, 100, -3, 3, 100, -3, 3)
            set_h2_style(pid_hs['cap_end'][pdg_id])
            pid_hs['cap_end'][pdg_id].Fill(y, x)
        else:
            pid_hs['cap_end'][pdg_id].Fill(y, x)

    pids = pid_hs['wall'].keys()
    for pid in pids:
        pdg_name = PDG.GetParticle(pid).GetName()
        h_wall = pid_hs['wall'][pid]
        try:
            h_cap_start = pid_hs['cap_start'][pid]
        except KeyError as e:
            # th2d_name = 'h_cap_start_{}'.format(pdg_name)
            # h_cap_start = TH2D(th2d_name, th2d_name, 100, -3, 3, 100, -3, 3)
            continue
        try:
            h_cap_end = pid_hs['cap_end'][pid]
        except KeyError as e:
            # th2d_name = 'h_cap_end_{}'.format(pdg_name)
            # h_cap_end = TH2D(th2d_name, th2d_name, 100, -3, 3, 100, -3, 3)
            continue

        # print('pid = {}'.format(pid))
        # print('pdg_name = {}'.format(pdg_name))
        # print('h_wall.Integral() = {}'.format(h_wall.Integral()))

        c1 = TCanvas('c1', 'c1', 1500, 800)
        set_margin()
        set_h2_color_style()
        gPad.SetBottomMargin(0.15)
        gPad.SetLeftMargin(0.15)

        c1.cd()
        pad1 = TPad("pad1", "pad1", 0, 0, 0.25, 1)
        pad1.SetTopMargin(0.36)
        pad1.SetBottomMargin(0.36)
        pad1.SetLeftMargin(0.2)
        pad1.SetRightMargin(0.2)
        pad1.Draw()
        pad1.cd()
        h_cap_start.Draw('colz')
        h_cap_start.GetXaxis().SetTitle('Y (m)')
        h_cap_start.GetYaxis().SetTitle('X (m)')
        h_cap_start.GetYaxis().SetTitleOffset(2.2)
        el_cap_start = TEllipse(0, 0, 3)
        el_cap_start.SetFillStyle(0)
        el_cap_start.Draw()

        c1.cd()
        pad2 = TPad("pad2", "pad2", 0.25, 0, 0.75, 1)
        pad2.SetTopMargin(0.1)
        pad2.SetBottomMargin(0.1)
        pad2.SetLeftMargin(0.15)
        pad2.SetRightMargin(0.12)
        pad2.Draw()
        pad2.cd()
        h_wall.Draw('colz')
        h_wall.GetXaxis().SetTitle('Z (m)')
        h_wall.GetYaxis().SetTitle('Angle from +Y-Axis (degree)')
        h_wall.GetYaxis().SetTitleOffset(1.8)

        c1.cd()
        pad3 = TPad("pad3", "pad3", 0.75, 0, 1, 1)
        pad3.SetTopMargin(0.36)
        pad3.SetBottomMargin(0.36)
        pad3.SetLeftMargin(0.2)
        pad3.SetRightMargin(0.2)
        pad3.Draw()
        pad3.cd()
        h_cap_end.Draw('colz')
        h_cap_end.GetXaxis().SetTitle('Y (m)')
        h_cap_end.GetYaxis().SetTitle('X (m)')
        h_cap_end.GetYaxis().SetTitleOffset(2.2)
        el_cap_end = TEllipse(0, 0, 3)
        el_cap_end.SetFillStyle(0)
        el_cap_end.Draw()

        c1.Update()
        c1.SaveAs('{}/plot_radiation_position.{}.{}.pdf'.format(FIGURE_DIR, filename, pdg_name))
        # input('Press any key to continue.')
        # break


def plot_radiation_momentum(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    pid_hs = {
        'wall': {},
        'cap_start': {},
        'cap_end': {}
    }

    x_max = 3

    for event in tf.Get('VirtualDetector/wall'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        h_name = 'h_wall_{}'.format(pdg_name)
        momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1000.

        if pdg_id not in pid_hs['wall']:
            pid_hs['wall'][pdg_id] = TH1D(h_name, h_name, 100, 0, x_max)
            set_h1_style(pid_hs['wall'][pdg_id])
            pid_hs['wall'][pdg_id].Fill(momentum)
        else:
            pid_hs['wall'][pdg_id].Fill(momentum)

    for event in tf.Get('VirtualDetector/cap_start'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        h_name = 'h_cap_start_{}'.format(pdg_name)
        momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1000.
        if pdg_id not in pid_hs['cap_start']:
            pid_hs['cap_start'][pdg_id] = TH1D(h_name, h_name, 100, 0, x_max)
            set_h1_style(pid_hs['cap_start'][pdg_id])
            pid_hs['cap_start'][pdg_id].Fill(momentum)
        else:
            pid_hs['cap_start'][pdg_id].Fill(momentum)

    for event in tf.Get('VirtualDetector/cap_end'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        h_name = 'h_cap_end_{}'.format(pdg_name)
        momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5 / 1000.
        if pdg_id not in pid_hs['cap_end']:
            pid_hs['cap_end'][pdg_id] = TH1D(h_name, h_name, 100, 0, x_max)
            set_h1_style(pid_hs['cap_end'][pdg_id])
            pid_hs['cap_end'][pdg_id].Fill(momentum)
        else:
            pid_hs['cap_end'][pdg_id].Fill(momentum)

    pids = pid_hs['wall'].keys()
    for pid in pids:
        pdg_name = PDG.GetParticle(pid).GetName()
        h_wall = pid_hs['wall'][pid]
        try:
            h_cap_start = pid_hs['cap_start'][pid]
        except KeyError:
            continue
        try:
            h_cap_end = pid_hs['cap_end'][pid]
        except KeyError:
            continue

        c1 = TCanvas('c1', 'c1', 1500, 300)
        set_margin()
        gPad.SetBottomMargin(0.15)
        gPad.SetLeftMargin(0.15)

        c1.cd()
        pad1 = TPad("pad1", "pad1", 0, 0, 0.25, 1)
        pad1.SetTopMargin(0.1)
        pad1.SetBottomMargin(0.25)
        pad1.SetLeftMargin(0.2)
        pad1.SetRightMargin(0.1)
        pad1.Draw()
        pad1.cd()
        h_cap_start.Draw('colz')
        h_cap_start.GetXaxis().SetTitle('Momentum (GeV)')
        h_cap_start.GetYaxis().SetTitle('Particle Count')
        h_cap_start.GetYaxis().SetTitleOffset(1.)
        pad1.Update()

        draw_statbox(h_cap_start, x1=0.45, y1=0.67, x2=0.95, y2=1.)
        if pdg_name == 'neutron':
            gPad.SetLogy()

        c1.cd()
        pad2 = TPad("pad2", "pad2", 0.25, 0, 0.75, 1)
        pad2.SetTopMargin(0.1)
        pad2.SetBottomMargin(0.25)
        pad2.SetLeftMargin(0.15)
        pad2.SetRightMargin(0.1)
        pad2.Draw()
        pad2.cd()
        h_wall.Draw('colz')
        h_wall.GetXaxis().SetTitle('Momentum (GeV)')
        h_wall.GetYaxis().SetTitle('Particle Count')
        h_wall.GetYaxis().SetTitleOffset(0.6)
        pad2.Update()
        draw_statbox(h_wall, x1=0.7, y1=0.67, x2=0.95, y2=1.)
        if pdg_name == 'neutron':
            gPad.SetLogy()

        c1.cd()
        pad3 = TPad("pad3", "pad3", 0.75, 0, 1, 1)
        pad3.SetTopMargin(0.1)
        pad3.SetBottomMargin(0.25)
        pad3.SetLeftMargin(0.2)
        pad3.SetRightMargin(0.1)
        pad3.Draw()
        pad3.cd()
        h_cap_end.Draw('colz')
        h_cap_end.GetXaxis().SetTitle('Momentum (GeV)')
        h_cap_end.GetYaxis().SetTitle('Particle Count')
        h_cap_end.GetYaxis().SetTitleOffset(1.)
        pad1.Update()
        draw_statbox(h_cap_end, x1=0.45, y1=0.67, x2=0.95, y2=1.)
        if pdg_name == 'neutron':
            gPad.SetLogy()

        c1.Update()
        c1.SaveAs('{}/plot_radiation_momentum.{}.{}.pdf'.format(FIGURE_DIR, filename, pdg_name))
        # input('Press any key to continue.')
        # break


def plot_radiation_count(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    pid_hs = {
        'wall': {},
        'cap_start': {},
        'cap_end': {}
    }

    for event in tf.Get('VirtualDetector/wall'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        th2d_name = 'h_wall_{}'.format(pdg_name)

        theta = atan(event.y / event.x)
        if event.x >= 0.:
            theta = pi / 2. - theta
        if event.x < 0:
            theta = -(pi / 2. + theta)
        theta = theta * 180. / pi
        z = event.z / 1000

        if pdg_id not in pid_hs['wall']:
            pid_hs['wall'][pdg_id] = TH2D(th2d_name, th2d_name, 5, -1, 20, 5, -180, 180)
            set_h2_style(pid_hs['wall'][pdg_id])
            pid_hs['wall'][pdg_id].Fill(z, theta)
        else:
            pid_hs['wall'][pdg_id].Fill(z, theta)

    for event in tf.Get('VirtualDetector/cap_start'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        th2d_name = 'h_cap_start_{}'.format(pdg_name)
        y = event.y / 1000.
        x = event.x / 1000.
        if pdg_id not in pid_hs['cap_start']:
            pid_hs['cap_start'][pdg_id] = TH2D(th2d_name, th2d_name, 5, -3, 3, 5, -3, 3)
            set_h2_style(pid_hs['cap_start'][pdg_id])
            pid_hs['cap_start'][pdg_id].Fill(y, x)
        else:
            pid_hs['cap_start'][pdg_id].Fill(y, x)

    for event in tf.Get('VirtualDetector/cap_end'):
        pdg_id = int(event.PDGid)
        pdg_name = PDG.GetParticle(pdg_id).GetName()
        th2d_name = 'h_cap_end_{}'.format(pdg_name)
        y = event.y / 1000.
        x = event.x / 1000.
        if pdg_id not in pid_hs['cap_end']:
            pid_hs['cap_end'][pdg_id] = TH2D(th2d_name, th2d_name, 5, -3, 3, 5, -3, 3)
            set_h2_style(pid_hs['cap_end'][pdg_id])
            pid_hs['cap_end'][pdg_id].Fill(y, x)
        else:
            pid_hs['cap_end'][pdg_id].Fill(y, x)

    pids = pid_hs['wall'].keys()
    for pid in pids:
        pdg_name = PDG.GetParticle(pid).GetName()
        h_wall = pid_hs['wall'][pid]
        try:
            h_cap_start = pid_hs['cap_start'][pid]
        except KeyError as e:
            continue
        try:
            h_cap_end = pid_hs['cap_end'][pid]
        except KeyError as e:
            continue

        c1 = TCanvas('c1', 'c1', 1500, 800)
        set_margin()
        set_h2_color_style()
        gPad.SetBottomMargin(0.15)
        gPad.SetLeftMargin(0.15)

        c1.cd()
        pad1 = TPad("pad1", "pad1", 0, 0, 0.25, 1)
        pad1.SetTopMargin(0.36)
        pad1.SetBottomMargin(0.36)
        pad1.SetLeftMargin(0.2)
        pad1.SetRightMargin(0.2)
        pad1.Draw()
        pad1.cd()
        h_cap_start.Draw('colz,text')
        h_cap_start.GetXaxis().SetTitle('Y (m)')
        h_cap_start.GetYaxis().SetTitle('X (m)')
        h_cap_start.GetYaxis().SetTitleOffset(2.2)
        h_cap_start.SetMarkerSize(2)
        h_cap_start.SetMarkerColor(kWhite)
        h_cap_start.GetZaxis().SetLabelSize(0)
        el_cap_start = TEllipse(0, 0, 3)
        el_cap_start.SetFillStyle(0)
        el_cap_start.SetLineColor(kRed)
        el_cap_start.Draw()
        gPad.Update()
        p_cap_start = h_cap_start.GetListOfFunctions().FindObject('palette')
        p_cap_start.SetX1NDC(1.2)
        p_cap_start.SetX2NDC(1.2)

        c1.cd()
        pad2 = TPad("pad2", "pad2", 0.25, 0, 0.75, 1)
        pad2.SetTopMargin(0.1)
        pad2.SetBottomMargin(0.1)
        pad2.SetLeftMargin(0.15)
        pad2.SetRightMargin(0.12)
        pad2.Draw()
        pad2.cd()
        h_wall.Draw('colz,text')
        h_wall.GetXaxis().SetTitle('Z (m)')
        h_wall.GetYaxis().SetTitle('Angle from +Y-Axis (degree)')
        h_wall.GetYaxis().SetTitleOffset(1.8)
        h_wall.SetMarkerSize(2)
        h_wall.SetMarkerColor(kWhite)
        gPad.Update()
        p_wall = h_wall.GetListOfFunctions().FindObject('palette')
        p_wall.SetX1NDC(1.2)
        p_wall.SetX2NDC(1.2)

        c1.cd()
        pad3 = TPad("pad3", "pad3", 0.75, 0, 1, 1)
        pad3.SetTopMargin(0.36)
        pad3.SetBottomMargin(0.36)
        pad3.SetLeftMargin(0.2)
        pad3.SetRightMargin(0.2)
        pad3.Draw()
        pad3.cd()
        h_cap_end.Draw('colz,text')
        h_cap_end.GetXaxis().SetTitle('Y (m)')
        h_cap_end.GetYaxis().SetTitle('X (m)')
        h_cap_end.GetYaxis().SetTitleOffset(2.2)
        h_cap_end.SetMarkerSize(2)
        h_cap_end.SetMarkerColor(kWhite)
        el_cap_end = TEllipse(0, 0, 3)
        el_cap_end.SetFillStyle(0)
        el_cap_end.SetLineColor(kRed)
        el_cap_end.Draw()
        gPad.Update()
        p_cap_end = h_cap_end.GetListOfFunctions().FindObject('palette')
        p_cap_end.SetX1NDC(1.2)
        p_cap_end.SetX2NDC(1.2)

        c1.Update()
        c1.SaveAs('{}/plot_radiation_count.{}.{}.pdf'.format(FIGURE_DIR, filename, pdg_name))
        # input('Press any key to continue.')
        # break


def print_radiation_tex(filename, momentum):
    pdg_names = ['neutron', 'mu-', 'mu+', 'pi-', 'pi+', 'proton']

    for pdg_name in pdg_names:
        print('\n% .........................................................\n')
        print('\\begin{frame}')
        print('  \\frametitle{{{}: Position and Momentum Distributions for {}}}'.format(momentum, pdg_name))
        print('  \\vspace{-3mm}')
        print('  \\begin{figure}')
        print('    \\includegraphics[width = \linewidth]{{figures/{{plot_radiation_position.{}.{}}}.pdf}} \\\\'.format(filename, pdg_name))
        print('    \\includegraphics[width = \linewidth]{{figures/{{plot_radiation_momentum.{}.{}}}.pdf}} \\\\'.format(filename, pdg_name))
        # print('    \\caption{{Position and momentum distributions for {}.}}'.format(pdg_name))
        print('  \\end{figure}')
        print('\\end{frame}')
        print('\n% .........................................................\n')
        print('\\begin{frame}')
        print('  \\frametitle{{{}: Particle Count by Region for {}}}'.format(momentum, pdg_name))
        print('  \\begin{figure}')
        print('    \\includegraphics[width = \linewidth]{{figures/{{plot_radiation_count.{}.{}}}.pdf}} \\\\'.format(filename, pdg_name))
        # print('    \\caption{{Particle count by region for {}.}}'.format(pdg_name))
        print('  \\end{figure}')
        print('\\end{frame}')


def print_radiation_summary(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    name_infos = {}

    for detector in ['wall', 'cap_start', 'cap_end']:
        for event in tf.Get('VirtualDetector/{}'.format(detector)):
            pdg_id = int(event.PDGid)
            name = PDG.GetParticle(pdg_id).GetName()
            momentum = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5
            if name not in name_infos:
                name_infos[name] = {
                    'count': 1,
                    'momentum': momentum
                }
            else:
                name_infos[name]['count'] += 1
                name_infos[name]['momentum'] += momentum

    for name in ['neutron', 'mu-', 'mu+', 'pi-', 'pi+', 'proton']:
        name_infos[name]['momentum'] /= name_infos[name]['count']
        print('{} & {} & {:.0f} \\\\'.format(name, name_infos[name]['count'], name_infos[name]['momentum']))


def get_radiation_length(Z, A):
    return 716.4 * A / (Z * (Z + 1) * log(287. / sqrt(Z))) # g/cm2
    
def print_radiation_length():
    Z_nitrogen = 7.
    A_nitrogen = 14.

    Z_oxygen = 8.
    A_oxygen = 16.

    Z_helium = 2.
    A_helium = 4.
    density_helium = 0.164e-3   # g/cm3
    air_density = 1.161e-3      # g/cm3
    air_nitrogen_mass_fraction = 0.76
    air_oxygen_mass_fraction = 0.24

    radiation_length_helium = get_radiation_length(Z_helium, A_helium)
    radiation_length_helium_cm = radiation_length_helium / density_helium

    radiation_length_air = 1. / (air_nitrogen_mass_fraction / get_radiation_length(Z_nitrogen, A_nitrogen) + air_oxygen_mass_fraction / get_radiation_length(Z_oxygen, A_oxygen))
    radiation_length_air_cm = radiation_length_air / air_density

    print('radiation_length_helium = {}'.format(radiation_length_helium))
    print('radiation_length_helium_cm = {}'.format(radiation_length_helium_cm))

    print('radiation_length_air = {}'.format(radiation_length_air))
    print('radiation_length_air_cm = {}'.format(radiation_length_air_cm))

    Z_carbon = 6
    A_carbon = 12
    density_carbon_dioxide = 1.799e-3 # g/cm3
    carbon_dioxide_carbon_mass_fraction = A_carbon / (A_carbon + A_oxygen * 2.)
    carbon_dioxide_oxygen_mass_fraction = A_oxygen * 2. / (A_carbon + A_oxygen * 2.)
    radiation_length_carbon_dioxide = 1. / (carbon_dioxide_carbon_mass_fraction / get_radiation_length(Z_carbon, A_carbon) + carbon_dioxide_oxygen_mass_fraction / get_radiation_length(Z_oxygen, A_oxygen))
    radiation_length_carbon_dioxide_cm = radiation_length_carbon_dioxide / density_carbon_dioxide
    print('radiation_length_carbon_dioxide = {}'.format(radiation_length_carbon_dioxide))
    print('radiation_length_carbon_dioxide_cm = {}'.format(radiation_length_carbon_dioxide_cm))

# 20180625_testbeam_64_32_16_8GeV_different_bs
# print_radiation_length()
# plot_particle_count_vs_secondary_beam_energy('gr_total', y_min=0, y_max=30)
# plot_particle_count_vs_secondary_beam_energy('gr_pi', y_min=0, y_max=15)
# plot_particle_count_vs_secondary_beam_energy('gr_proton', y_min=0, y_max=15)
# plot_particle_momentum('g4bl.b_-1.8T.pi+.64000.csv', 1500, 3500, title='64 GeV Secondary Beam', y_max=0., bin_count=20, y_title_offset=1.4, normalization_factor=4., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.8T.pi+.32000.csv', 1500, 3500, title='32 GeV Secondary Beam', y_max=0., bin_count=20, y_title_offset=1.4, normalization_factor=12., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.8T.pi+.16000.csv', 1500, 3500, title='16 GeV Secondary Beam', y_max=0., bin_count=20, y_title_offset=1.4, normalization_factor=24.32, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.8T.pi+.8000.csv', 1500, 3500, title='8 GeV Secondary Beam', y_max=0., bin_count=20, y_title_offset=1.4, normalization_factor=103.8, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.35T.pi+.64000.csv', 1200, 2500, title='64 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=4., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.35T.pi+.32000.csv', 1200, 2500, title='32 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=12., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.35T.pi+.16000.csv', 1200, 2500, title='16 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=28.76, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-1.35T.pi+.8000.csv', 1200, 2500, title='8 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=103.7, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-0.45T.pi+.64000.csv', 100, 1400, title='64 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=4., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-0.45T.pi+.32000.csv', 100, 1400, title='32 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=12., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-0.45T.pi+.16000.csv', 100, 1400, title='16 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=24.5, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-0.45T.pi+.8000.csv', 100, 1400, title='8 GeV Secondary Beam', y_max=0., bin_count=13, y_title_offset=1.4, normalization_factor=103.75, y_title='Particle Count per 1M Beam Particles')
plot_particle_momentum('beamline.py.in.job_1_1800.18m.b_-0.9T.pi+_64gev.root.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=18., y_title='Particle Count per 1M Beam Particles')
plot_particle_momentum('beamline.py.in.job_1_1800.27m.b_-0.9T.pi+_32gev.root.csv', 700, 1800, title='32 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=27., y_title='Particle Count per 1M Beam Particles')
plot_particle_momentum('beamline.py.in.job_1_900.45m.b_-0.9T.pi+_16gev.root.csv', 700, 1800, title='16 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=45., y_title='Particle Count per 1M Beam Particles')
plot_particle_momentum('beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root.csv', 700, 1800, title='8 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=90., y_title='Particle Count per 1M Beam Particles')
# get_particle_count_vs_secondary_beam_energy(suffix='b_-1.8T', csv_64gev='g4bl.b_-1.8T.pi+.64000.csv', norm_64gev=4., csv_32gev='g4bl.b_-1.8T.pi+.32000.csv', norm_32gev=12., csv_16gev='g4bl.b_-1.8T.pi+.16000.csv', norm_16gev=24.32, csv_8gev='g4bl.b_-1.8T.pi+.8000.csv', norm_8gev=103.8,)
# get_particle_count_vs_secondary_beam_energy(suffix='b_-1.35T', csv_64gev='g4bl.b_-1.35T.pi+.64000.csv', norm_64gev=4., csv_32gev='g4bl.b_-1.35T.pi+.32000.csv', norm_32gev=12., csv_16gev='g4bl.b_-1.35T.pi+.16000.csv', norm_16gev=28.76, csv_8gev='g4bl.b_-1.35T.pi+.8000.csv', norm_8gev=103.7,)
# get_particle_count_vs_secondary_beam_energy(suffix='b_-0.45T', csv_64gev='g4bl.b_-0.45T.pi+.64000.csv', norm_64gev=4., csv_32gev='g4bl.b_-0.45T.pi+.32000.csv', norm_32gev=12., csv_16gev='g4bl.b_-0.45T.pi+.16000.csv', norm_16gev=24.5, csv_8gev='g4bl.b_-0.45T.pi+.8000.csv', norm_8gev=103.75,)
# get_particle_count_vs_secondary_beam_energy(suffix='b_-0.9T', csv_64gev='beamline.py.in.job_1_1800.18m.b_-0.9T.pi+_64gev.root.csv', norm_64gev=18., csv_32gev='beamline.py.in.job_1_1800.27m.b_-0.9T.pi+_32gev.root.csv', norm_32gev=27., csv_16gev='beamline.py.in.job_1_900.45m.b_-0.9T.pi+_16gev.root.csv', norm_16gev=45., csv_8gev='beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root.csv', norm_8gev=90.,)

# 20180530_testbeam_radiation_dosage
# gStyle.SetOptStat(0)
# plot_radiation_position('radiation.10000.64GeV.root')
# gStyle.SetOptStat('emr')
# plot_radiation_momentum('radiation.10000.64GeV.root')
# gStyle.SetOptStat(0)
# gStyle.SetPalette(51)
# TGaxis.SetMaxDigits(2)
# plot_radiation_count('radiation.10000.64GeV.root')
# print_radiation_tex('radiation.10000.64GeV.root', '64 GeV')
# gStyle.SetOptStat(0)
# plot_radiation_position('radiation.10000.32GeV.root')
# gStyle.SetOptStat('emr')
# plot_radiation_momentum('radiation.10000.32GeV.root')
# gStyle.SetOptStat(0)
# plot_radiation_count('radiation.10000.32GeV.root')
# print_radiation_tex('radiation.10000.32GeV.root', '32 GeV')
# print_radiation_summary('radiation.10000.64GeV.root')
# print_radiation_summary('radiation.10000.32GeV.root')

# test_beam_neutrino_2018, poster
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.45T.10m.root.csv', 350, 800, y_max=0.8, bin_count=15, y_title_offset=1.4, normalization_factor=9, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.9T.10m.root.csv', 800, 1500, y_max=1.5, bin_count=15, y_title_offset=1.4, normalization_factor=9, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.35T.10m.root.csv', 1200, 2400, y_max=1.2, bin_count=15, y_title_offset=1.4, normalization_factor=9, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.8T.10m.root.csv', 1600, 3000, y_max=0.8, bin_count=15, y_title_offset=1.4, normalization_factor=9, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.45T.10m.root.csv', 350, 800, y_max=7, bin_count=15, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.9T.10m.root.csv', 800, 1500, y_max=15, bin_count=15, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.35T.10m.root.csv', 1200, 2400, y_max=10, bin_count=15, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.8T.10m.root.csv', 1600, 3200, y_max=8, bin_count=15, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count')

# 20180509_testbeam_64_32_16_8GeV
# gStyle.SetOptStat(0)
# plot_trigger_particle('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root.csv', show_boundary=True)
# plot_noise_particle('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root.csv', show_boundary=True)
# save_particle_to_csv('beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root')
# plot_particle_momentum('beamline.py.in.job_1_1800.18m.b_-0.9T.pi+_64gev.root.csv', 800, 1500, y_max=15, normalization_factor=1.8, y_title='Particle Count per 10M Secondary Beam Particles')
# plot_particle_momentum('beamline.py.in.job_1_1800.27m.b_-0.9T.pi+_32gev.root.csv', 800, 1500, y_max=10, normalization_factor=2.7, y_title='Particle Count per 10M Secondary Beam Particles')
# plot_particle_momentum('beamline.py.in.job_1_900.45m.b_-0.9T.pi+_16gev.root.csv', 800, 1500, y_max=6, normalization_factor=4.5, y_title='Particle Count per 10M Secondary Beam Particles')
# plot_particle_momentum('beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root.csv', 800, 1500, y_max=5, normalization_factor=9., y_title='Particle Count per 10M Secondary Beam Particles')
# plot_particle_count_vs_secondary_beam_energy()

# 20180413_testbeam_120gev
# save_particle_to_csv('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.proton.root')
# save_particle_to_csv('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root')
# plot_particle_momentum('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.proton.root.csv', 800, 1500, y_max=25)
# plot_particle_momentum('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root.csv', 800, 1500, y_max=15)
# plot_particle_momentum('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.proton.root.csv', 0, 3000, log_y=True, y_max=500000, plot_noise=True)
# plot_particle_momentum('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root.csv', 0, 3000, log_y=True, y_max=500000, plot_noise=True)
# gStyle.SetOptStat(0)
# plot_noise_particle('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.proton.root.csv')
# plot_noise_particle('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root.csv')
# compare_particle_count()
# plot_trigger_particle('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.proton.root.csv')
# plot_trigger_particle('beamline.py.in.job_1_900.10k_per_job.b_-0.9T.pi+.root.csv')

# 20180318_testbeam_new_setup
# plot_time_of_flight(distance=12.8, y_min=3.e4, y_max=5.e5, canvas_height=600)
# plot_time_of_flight_diff(distance=12.8, y_max=5e6, canvas_height=600)
# plot_time_of_flight_mc(distance=6.075)
# plot_time_of_flight_mc(distance=12.8)
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.10m.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.9T.10m.root')
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.10m.root.csv', 300, 1000, 22)
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.9T.10m.root.csv', 800, 2000, 10)
# plot_particle_angle('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.10m.root.csv')
# plot_particle_angle('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.9T.10m.root.csv')
# save_particle_to_csv('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.45T.10m.root')
# save_particle_to_csv('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.9T.10m.root')
# save_particle_to_csv('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.35T.10m.root')
# save_particle_to_csv('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.8T.10m.root')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.45T.10m.root.csv', 350, 800, 7, bin_count=15)
# plot_particle_angle('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.45T.10m.root.csv')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.9T.10m.root.csv', 800, 1500, 15, bin_count=15)
# plot_particle_angle('beam.py.in.30_spill.job_1_900.10k_per_job.b_-0.9T.10m.root.csv')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.35T.10m.root.csv', 1200, 2200, 12, bin_count=15)
# plot_particle_angle('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.35T.10m.root.csv')
# plot_particle_momentum('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.8T.10m.root.csv', 1600, 3000, 8, bin_count=15)
# plot_particle_angle('beam.py.in.30_spill.job_1_900.10k_per_job.b_-1.8T.10m.root.csv')

# 20180309_testbeam_cherenkov
# plot_cherenkov_index_of_refaction()
# get_cherenkov_photon_count()

# 20180308_testbeam_kalman_filter
# test_1d_kalman()
# test_1d_kalman_prediction_only()
# test_graph_shade()

# 20180211_testbeam_high_stat
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_0.45T.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_1.8T.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_-1.8T.root')
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_0.45T.root.csv', 300, 2000, 20)
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_1.8T.root.csv', 2000, 6000, 10)
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.root.csv', 300, 2000, 22)
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_-1.8T.root.csv', 2000, 6000, 10)

# 20180123_testbeam_cu_target
# plot_pxy_thetas('target.64GeV.root')
# plot_pxy_thetas('target.32GeV.root')
# plot_pxy_thetas('target.16GeV.root')
# plot_pxy_thetas('target.8GeV.root')
# plot_pxy_thetas('target.8GeV.root')
# plot_momentum_pxy_thetas()
# print_slide_momentum_pxy_thetas()

# 20180118_testbeam_m1_magnet
# compute_bending_angle()
# compute_b_times_l()
# get_min_momentum()
# plot_m1_upstream()
# plot_m1_downstream()
# plot_m1_block_momentum()
# plot_p_vs_angle_max_angle()
# plot_max_theta()
# plot_min_b_field()

# 20180109_testbeam_momentum_pid
# plot_p_vs_angle()
# plot_cherenkov()
# plot_time_of_flight()
# plot_time_of_flight_diff()

# 20171211_test_beam_geometry
# get_particle_count_filter()
# get_particle_count()
# print_particle_count_table()
# generate_text()
# print(get_momentum(237.843, 938.272))
# plot_momentum()
