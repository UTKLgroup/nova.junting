from rootalias import *
from util import *
from pprint import pprint
import csv
import math
from math import pi, cos, sin, atan, sqrt, log, exp
import numpy as np
from datetime import datetime
import os


PDG = TDatabasePDG()
SPEED_OF_LIGHT = 3.e8              # m/s
ELEMENTARY_CHARGE = 1.60217662e-19 # coulomb
INCH_TO_METER = 2.54 / 100.
DEGREE_TO_RADIAN = 3.14 / 180.
RADIAN_TO_DEGREE = 180. / 3.14
COLORS = [kBlack, kBlue, kRed + 1, kMagenta + 2, kGreen + 1, kOrange + 1, kYellow + 2, kPink, kViolet, kAzure + 4, kCyan + 1, kTeal - 7, kBlue - 5]
FIGURE_DIR = '/Users/juntinghuang/beamer/20190531_testbeam_good_particle_position/figures'
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
    colors = [kRed + 1, kMagenta + 2, kBlue + 1, kGreen + 2, kBlack]
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


def save_particle_momentum_csv(filename, x_min, x_max, **kwargs):
    bin_count = kwargs.get('bin_count', 50)
    normalization_factor = kwargs.get('normalization_factor', 1.)

    h_all = TH1D('h_all', 'h_all', bin_count, x_min, x_max)
    pid_hists = {}
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

            h_all.Fill(momentum)
            if pid not in pid_hists:
                pid_hists[pid] = TH1D('h_{}'.format(pid), 'h_{}'.format(pid), bin_count, x_min, x_max)
                pid_hists[pid].Fill(momentum)
            else:
                pid_hists[pid].Fill(momentum)

    tf_out = TFile('{}/{}.hist.root'.format(DATA_DIR, filename), 'RECREATE')
    for pid, hist in pid_hists.items():
        hist.Scale(1. / normalization_factor)
        hist.Write('h_{}'.format(pid))
    h_all.Scale(1. / normalization_factor)
    h_all.Write('h_all')
    tf_out.Close()


def save_particle_momentum_root(filename, x_min, x_max, **kwargs):
    bin_count = kwargs.get('bin_count', 50)
    normalization_factor = kwargs.get('normalization_factor', 1.)
    noise_particle = kwargs.get('noise_particle', False)

    h_all = TH1D('h_all', 'h_all', bin_count, x_min, x_max)
    pid_hists = {}

    tf_in = TFile('{}/{}'.format(DATA_DIR, filename))
    tree = tf_in.Get('tree')
    particle_count_total = tree.GetEntries()
    particle_count = 0
    for particle in tree:
        particle_count += 1
        if particle_count % 1e6 == 0:
            print('particle_count = {} / {} ({:.1f}%)'.format(particle_count, particle_count_total, particle_count / particle_count_total * 100.))

        if not noise_particle and particle.is_noise:
            continue
        if noise_particle and not particle.is_noise:
            continue

        pid = int(particle.pdg_id)
        px = particle.px
        py = particle.py
        pz = particle.pz
        momentum = (px**2 + py**2 + pz**2)**0.5
        h_all.Fill(momentum)
        if pid not in pid_hists:
            pid_hists[pid] = TH1D('h_{}'.format(pid), 'h_{}'.format(pid), bin_count, x_min, x_max)
            pid_hists[pid].Fill(momentum)
        else:
            pid_hists[pid].Fill(momentum)

    tf_out = TFile('{}/{}.noise_particle_{}.hist.root'.format(DATA_DIR, filename, noise_particle), 'RECREATE')
    for pid, hist in pid_hists.items():
        hist.Scale(1. / normalization_factor)
        hist.Write('h_{}'.format(pid))
    h_all.Scale(1. / normalization_factor)
    h_all.Write('h_all')
    tf_out.Close()


def plot_particle_momentum(filename, x_min, x_max, **kwargs):
    bin_count = kwargs.get('bin_count', 50)
    y_max = kwargs.get('y_max', 0.)
    y_title = kwargs.get('y_title', 'Particle Count')
    log_y = kwargs.get('log_y', False)
    plot_noise = kwargs.get('plot_noise', False)
    normalization_factor = kwargs.get('normalization_factor', 1.)
    y_title_offset = kwargs.get('y_title_offset', 1.8)
    b_field = kwargs.get('b_field', None)
    beam_momentum = kwargs.get('beam_momentum', 64)

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
        # lg1 = TLegend(0.51, 0.6, 0.84, 0.84)
        # lg1 = TLegend(0.49, 0.6, 0.82, 0.84)
        lg1 = TLegend(0.47, 0.6, 0.80, 0.84)
    set_legend_style(lg1)

    pid_counts = sorted(pid_counts, key=lambda x: x[1], reverse=True)
    pids = [pid_count[0] for pid_count in pid_counts]
    for i, pid in enumerate(pids):
        hist = pid_hists[pid]
        set_h1_style(hist)
        hist.SetLineColor(colors[i])

        if i == 0:
            hist.Draw('hist')
            hist.GetXaxis().SetTitle('Momentum (MeV)')
            hist.GetYaxis().SetTitle(y_title)
            hist.GetYaxis().SetTitleOffset(y_title_offset)
            hist.SetTitle('{} GeV Beam, B = {:.1f} T'.format(beam_momentum, b_field))
            if y_max:
                hist.GetYaxis().SetRangeUser(0 if not log_y else 0.5, y_max)
        else:
            hist.Draw('hist,sames')

        if (PDG.GetParticle(pid).Charge() > 0. and b_field < 0.) or (PDG.GetParticle(pid).Charge() < 0. and b_field > 0.):
            if plot_noise:
                lg1.AddEntry(hist, '{} ({:.0f}, {:.0f} MeV)'.format(PDG.GetParticle(pid).GetName(), pid_counts[i][1] / normalization_factor, hist.GetMean()), 'l')
            else:
                lg1.AddEntry(hist, '{1} ({2:.{0}f})'.format(count_precision, PDG.GetParticle(pid).GetName(), pid_counts[i][1] / normalization_factor), 'l')
        else:
            print('Wrong sign particles: pid = {}, count = {}, avg momentum = {}'.format(pid, pid_counts[i][1], sum(pid_momentums[pid]) / len(pid_momentums[pid])))

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(43)
    latex.SetTextSize(28)
    # latex.DrawLatex(0.2, 0.83, label)

    x_latex = 0.49
    if not plot_noise:
        latex.DrawLatex(x_latex, 0.42, 'rms = {:.0f} MeV'.format(h_all.GetRMS()))
        latex.DrawLatex(x_latex, 0.48, 'mean = {:.0f} MeV'.format(h_all.GetMean()))
        latex.DrawLatex(x_latex, 0.54, 'total count = {1:.{0}f}'.format(count_precision, h_all.Integral()))
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_particle_momentum.{}.plot_noise_{}.normalization_factor_{}.pdf'.format(FIGURE_DIR, filename, plot_noise, normalization_factor))
    input('Press any key to continue.')


def plot_saved_particle_momentum(filename, **kwargs):
    rebin = kwargs.get('rebin', 0)
    log_y = kwargs.get('log_y', False)
    y_max = kwargs.get('y_max', 0.)
    y_min = kwargs.get('y_min', 0.001)
    x_min = kwargs.get('x_min', 0.)
    x_max = kwargs.get('x_max', 3500.)
    y_title = kwargs.get('y_title', 'Particle Count per 1M Beam Particles')
    y_title_offset = kwargs.get('y_title_offset', 1.5)
    b_field = kwargs.get('b_field', None)
    beam_momentum = kwargs.get('beam_momentum', 64)
    noise_particle = kwargs.get('noise_particle', False)

    pid_hists = {}
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_all = tf.Get('h_all')
    if rebin:
        h_all.Rebin(rebin)
    for key in tf.GetListOfKeys():
        hist_name = key.GetName()
        if hist_name == 'h_all':
            continue
        pid = int(hist_name.split('_')[1])
        pid_hists[pid] = tf.Get(hist_name)
        if rebin:
            pid_hists[pid].Rebin(rebin)

    if y_max == 0.:
        for pid, hist in pid_hists.items():
            if hist.GetMaximum() > y_max:
                y_max = hist.GetMaximum()
        if log_y:
            y_max *= 5
        else:
            y_max *= 1.2

    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    gStyle.SetOptStat(0)
    if log_y:
        gPad.SetLogy()

    colors = [
        kBlack,
        kRed,
        kBlue,
        kMagenta + 1,
        kGreen + 1,
        kViolet + 1,
        kYellow + 2,
        kOrange + 2,
        kAzure - 7,
        kCyan + 2,
        kPink - 8,
        kSpring + 9,
        kTeal + 2
    ]

    lg1 = TLegend(0.6, 0.6, 0.76, 0.84)
    set_legend_style(lg1)
    lg1.SetTextSize(24)
    if noise_particle:
        lg1.SetNColumns(2)
        lg1.SetX1(0.24)
        lg1.SetX2(0.88)
        lg1.SetY1(0.66)
        lg1.SetY2(0.87)
        lg1.SetTextSize(20)
        # c1.SetCanvasSize(800, 800);

    pids = []
    for pid in pid_hists.keys():
        if not noise_particle and (PDG.GetParticle(pid).Charge() == 0 or (PDG.GetParticle(pid).Charge() < 0. and b_field < 0.) or (PDG.GetParticle(pid).Charge() > 0. and b_field > 0.)):
            print('Wrong sign particles: pid = {}, count = {}, avg momentum = {}'.format(pid, pid_hists[pid].Integral(), pid_hists[pid].GetMean()))
            continue
        pids.append(pid)
    pids = sorted(pids, key=lambda x: (abs(x), np.sign(x)))

    for i, pid in enumerate(pids):
        hist = pid_hists[pid]
        set_h1_style(hist)
        hist.SetLineColor(colors[i])

        if i == 0:
            hist.Draw('hist')
            hist.GetXaxis().SetTitle('Momentum (MeV)')
            hist.GetYaxis().SetTitle(y_title)
            hist.GetYaxis().SetTitleOffset(y_title_offset)
            hist.SetTitle('{} GeV Beam, B = {:.3f} T'.format(beam_momentum, b_field))
            hist.GetYaxis().SetRangeUser(0 if not log_y else y_min, y_max)
            hist.GetXaxis().SetRangeUser(x_min, x_max)
        else:
            hist.Draw('hist,sames')
        lg1.AddEntry(hist, '{0} ({1:.1E})'.format(PDG.GetParticle(pid).GetName(), hist.Integral()), 'l')

    lg1.Draw()
    if not noise_particle:
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextFont(43)
        latex.SetTextSize(24)
        x_latex = 0.61
        latex.DrawLatex(x_latex, 0.42, 'rms = {:.0f} MeV'.format(h_all.GetRMS()))
        latex.DrawLatex(x_latex, 0.48, 'mean = {:.0f} MeV'.format(h_all.GetMean()))
        latex.DrawLatex(x_latex, 0.54, 'total count = {0:.1f}'.format(h_all.Integral()))

    c1.Update()
    c1.SaveAs('{}/plot_saved_particle_momentum.{}.pdf'.format(FIGURE_DIR, filename))
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
    colors = [kBlue + 1, kGreen + 2, kRed + 1]
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    eta = 4.1e-4                  # atm-1

    momentums = np.arange(0.01, 10, 0.01)
    rrefraction_indexs = []
    for i, mass in enumerate(masses):
        refraction_indexs = []
        for momentum in momentums:
            # pressure = 1. / eta * ((1 + (mass / momentum)**2)**0.5 - 1.)
            refraction_index = (1 + (mass / momentum)**2)**0.5
            refraction_indexs.append(refraction_index)
        rrefraction_indexs.append(refraction_indexs)

    grs = []
    for i in range(len(rrefraction_indexs)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(rrefraction_indexs[i]))
        set_graph_style(gr)
        grs.append(gr)

    refraction_index_one_atm = eta * 1. + 1.;
    one_atm_refraction_indexs = [refraction_index_one_atm for i in range(len(momentums))]
    gr_one_atm = TGraph(len(momentums), np.array(momentums), np.array(one_atm_refraction_indexs))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    # gPad.SetLogy()
    gPad.SetGrid()
    gPad.SetLeftMargin(0.2)

    # lg1 = TLegend(0.2, 0.8, 0.88, 0.88)
    # lg1 = TLegend(0.3, 0.18, 0.8, 0.26)
    lg1 = TLegend(0.23, 0.62, 0.58, 0.86)
    set_legend_style(lg1)
    # lg1.SetNColumns(5)

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

    set_graph_style(gr_one_atm)
    gr_one_atm.SetLineStyle(2)
    gr_one_atm.Draw('L')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(43)
    latex.SetTextSize(28)
    latex.DrawLatex(0.5, 0.25, '1 atm')

    c1.Update()
    c1.SaveAs('{}/plot_cherenkov_index_of_refaction.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_cherenkov_photon_count():
    index_of_refraction = 1.0004 # CO2
    # index_of_refraction = 1.000035 # helium
    # index_of_refraction = 1.0003 # N2
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
    e_counts = []
    mu_counts = []
    k_counts = []

    total_count_errs = []
    pi_count_errs = []
    proton_count_errs = []
    e_count_errs = []
    mu_count_errs = []
    k_count_errs = []

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

        e_count = 0
        if -11 in pid_count:
            e_count = pid_count[-11]
        e_counts.append(e_count / norm)
        e_count_errs.append(sqrt(e_count) / norm)

        mu_count = 0
        if -13 in pid_count:
            mu_count = pid_count[-13]
        mu_counts.append(mu_count / norm)
        mu_count_errs.append(sqrt(mu_count) / norm)

        k_count = 0
        if 321 in pid_count:
            k_count = pid_count[321]
        k_counts.append(k_count / norm)
        k_count_errs.append(sqrt(k_count) / norm)

    gr_total = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(total_counts), np.array(beam_energy_errs), np.array(total_count_errs))
    gr_pi = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(pi_counts), np.array(beam_energy_errs), np.array(pi_count_errs))
    gr_proton = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(proton_counts), np.array(beam_energy_errs), np.array(proton_count_errs))
    gr_e = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(e_counts), np.array(beam_energy_errs), np.array(e_count_errs))
    gr_mu = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(mu_counts), np.array(beam_energy_errs), np.array(mu_count_errs))
    gr_k = TGraphErrors(len(beam_energies), np.array(beam_energies), np.array(k_counts), np.array(beam_energy_errs), np.array(k_count_errs))

    print('proton_counts = {}'.format(proton_counts))
    print('pi_counts = {}'.format(pi_counts))
    print('e_counts = {}'.format(e_counts))
    print('mu_counts = {}'.format(mu_counts))
    print('k_counts = {}'.format(k_counts))
    print('total_counts = {}'.format(total_counts))

    grs = [gr_total, gr_pi, gr_proton, gr_e, gr_mu, gr_k]
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
    gr_e.Write('gr_e')
    gr_mu.Write('gr_mu')
    gr_k.Write('gr_k')
    f_out.Close()
    print('Graphs saved to {}.'.format(filename))

    return gr_total, gr_pi, gr_proton, gr_e, gr_mu, gr_k


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
        tf = TFile('{}/{}'.format(DATA_DIR, filename))
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


def plot_birks_law():
    # dedxs = np.arange(0, 20, 0.1)
    # birks_constant = 0.0125     # g / MeV / cm2
    dedxs = np.arange(0, 7, 0.1)
    birks_constant = 0.01155    # g / MeV / cm2
    density = 0.862             # g / cm3
    chou_constant = 0.

    print('birks_constant / density = {}'.format(birks_constant / density))

    coeffs = []
    ones = []
    for dedx in dedxs:
        coeffs.append(1. / (1. + birks_constant / density * dedx + chou_constant * dedx**2))
        ones.append(1.)

    gr = TGraph(len(dedxs), np.array(dedxs), np.array(coeffs))
    gr_one = TGraph(len(dedxs), np.array(dedxs), np.array(ones))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_graph_style(gr)
    set_graph_style(gr_one)

    gr.GetXaxis().SetTitle('dE/dx (MeV / cm)')
    # gr.GetYaxis().SetTitle('1 / [1 + B(dE/dx) + C(dE/dx)^{2}]')
    gr.GetYaxis().SetTitle('1 / [1 + B(dE/dx)]')
    gr.Draw('AL')

    c1.Update()
    c1.SaveAs('{}/plot_birks_law.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_rindex(wavelength):
    # wavelength in nm
    return 1.45689 + 4362 / wavelength**2


def plot_cherenkov_photon_count():
    wavelength_low = 200
    wavelength_high = 400

    h_cherenkov_2d = TH2D('h_cherenkov_2d','h_cherenkov_2d', 2000, 0, 1, 100, wavelength_low, wavelength_high)
    for i_beta in range(1, h_cherenkov_2d.GetXaxis().GetNbins() + 1):
        for i_wavelength in range(1, h_cherenkov_2d.GetYaxis().GetNbins() + 1):
            beta = h_cherenkov_2d.GetXaxis().GetBinCenter(i_beta)
            wavelength = h_cherenkov_2d.GetYaxis().GetBinCenter(i_wavelength)
            delta_wavelength = h_cherenkov_2d.GetYaxis().GetBinWidth(i_wavelength)
            rindex = get_rindex(wavelength);
            photon_count = 0.0459 / wavelength**2 * (1. - 1. / (beta * rindex)**2) * delta_wavelength
            if photon_count < 0.:
                photon_count = 0.
            photon_count *= 1.e7 # change unit from per nm to per cm
            h_cherenkov_2d.SetBinContent(i_beta, i_wavelength, photon_count)

    h_cherenkov_beta = h_cherenkov_2d.ProjectionX('h_cherenkov_beta')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h2_color_style()
    set_h2_style(h_cherenkov_2d)
    gPad.SetLogz()
    gPad.SetRightMargin(0.2)
    h_cherenkov_2d.GetXaxis().SetRangeUser(0.5, 1)
    h_cherenkov_2d.GetXaxis().SetTitle('#beta')
    h_cherenkov_2d.GetYaxis().SetTitle('#lambda (nm)')
    h_cherenkov_2d.GetZaxis().SetTitle('Photon Count / cm')
    h_cherenkov_2d.Draw('colz')
    c1.Update()
    c1.SaveAs('{}/plot_cherenkov_photon_count.2d.png'.format(FIGURE_DIR))

    c2 = TCanvas('c2', 'c2', 800, 600)
    set_margin()
    set_h1_style(h_cherenkov_beta)
    h_cherenkov_beta.GetXaxis().SetRangeUser(0.5, 1)
    h_cherenkov_beta.GetXaxis().SetTitle('#beta')
    h_cherenkov_beta.GetYaxis().SetTitle('Photon Count / cm')
    h_cherenkov_beta.Draw()
    c2.Update()
    c2.SaveAs('{}/plot_cherenkov_photon_count.1d.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_dt_dz_collection_rate():
    tf = TFile('{}/dT_dZ_CollectionRate.root'.format(DATA_DIR))
    h1 = tf.Get('dT_dZ_CollectionRate')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h2_color_style()
    gPad.SetLogz()
    set_h2_style(h1)

    # h1.GetZaxis().SetRangeUser(1e-7, 1e-5)
    h1.GetXaxis().SetTitle('#Delta Z (cm)')
    h1.GetYaxis().SetTitle('#Delta T (ns)')
    h1.GetZaxis().SetTitle('Collection Fraction')
    h1.Draw('colz')

    c1.Update()
    c1.SaveAs('{}/plot_dt_dz_collection_rate.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_fiber_brightness():
    tf = TFile('{}/ndBrightnessFromCosmics.root'.format(DATA_DIR))
    h1 = tf.Get('BrightnessByCell')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h2_color_style()
    # gPad.SetLogz()
    set_h2_style(h1)

    # h1.GetZaxis().SetRangeUser(1e-7, 1e-5)
    # h1.GetXaxis().SetTitle('#Delta Z (cm)')
    # h1.GetYaxis().SetTitle('#Delta T (ns)')
    h1.GetZaxis().SetTitle('Fiber Brightness')
    h1.Draw('colz')

    c1.Update()
    c1.SaveAs('{}/plot_brightness.pdf'.format(FIGURE_DIR))
    c1.SaveAs('{}/plot_brightness.png'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_fiber_attenuation():
    xs = np.arange(0, 1600, 20.)
    ts = []
    for x in xs:
        ts.append(0.2667 * exp(-x / 254.) + 0.2139 * exp(-x / 860.))
        # ts.append(0.555 * exp(-x / 254.) + 0.445 * exp(-x / 860.))
    gr = TGraph(len(xs), np.array(xs), np.array(ts))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_graph_style(gr)
    gPad.SetGrid()

    gr.GetXaxis().SetTitle('Distance (cm)')
    gr.GetYaxis().SetTitle('Transmission')
    gr.GetYaxis().SetRangeUser(0, 0.6)
    gr.Draw('AL')

    c1.Update()
    c1.SaveAs('{}/plot_attenuation.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_dcs_threshold():
    tf = TFile('det/fd_thresholdDist_lowThresh.root')
    h2 = tf.Get('thresholds')
    h1 = h2.Clone()

    set_h1_style(h1)

    c1 = TCanvas('c1', 'c1', 800, 400)
    set_margin()
    gPad.SetBottomMargin(0.2)

    h1.GetXaxis().SetTitle('FD DCS threshold (ADC)')
    h1.GetYaxis().SetTitle('APD Pixel Count')
    h1.GetXaxis().SetTitleOffset(1.1)
    h1.GetYaxis().SetTitleOffset(1.1)
    h1.GetXaxis().SetRangeUser(30, 70)
    h1.Draw()
    c1.Update()
    draw_statbox(h1, y1=0.7, x1=0.7)

    c1.Update()
    c1.SaveAs('{}/plot_dcs_threshold.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_energy_loss_vs_cherenkov_length():
    lengths = [1., 1.5, 2., 2.5, 3., 3.5]
    air_energy_losses = [17.23, 17.28, 17.38, 17.41, 17.51, 17.54]
    helium_energy_losses = [12.55, 12.73, 12.96, 13.17, 13.37, 13.56]

    gr = TGraph(len(lengths), np.array(lengths), np.array(helium_energy_losses))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_graph_style(gr)
    gPad.SetGrid()

    gr.GetXaxis().SetTitle('Length of Cherenkov Counter (m)')
    gr.GetYaxis().SetTitle('Energy Loss in Beamline (MeV)')
    gr.Draw('AL')

    c1.Update()
    c1.SaveAs('{}/plot_energy_loss_vs_cherenkov_length.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def save_momentum_collimator_up():
    tf = TFile('data/beamline.py.radiation.collimator.in.pi+_85gev.15m.root')

    particle_count = 0
    pid_h_momentums = {}
    h_total = TH1D('h_total', 'h_total', 100000, 0, 100000)
    for particle in tf.Get('VirtualDetector/det'):
        particle_count += 1
        if particle_count % 100000 == 0:
            print('particle_count = {}'.format(particle_count))

        pid = int(particle.PDGid)
        px = particle.Px
        py = particle.Py
        pz = particle.Pz
        mass = PDG.GetParticle(pid).Mass()
        momentum = (px**2 + py**2 + pz**2)**0.5
        energy = (mass**2 + px**2 + py**2 + pz**2)**0.5

        if pid not in pid_h_momentums:
            pid_h_momentums[pid] = TH1D('h_{}'.format(pid), 'h_{}'.format(pid), 100000, 0, 100000)
        # pid_h_momentums[pid].Fill(energy)
        # h_total.Fill(energy)
        pid_h_momentums[pid].Fill(momentum)
        h_total.Fill(momentum)

    tf_out = TFile('data/beamline.py.radiation.collimator.in.pi+_85gev.15m.hist.root', 'RECREATE')
    for pid in pid_h_momentums:
        pid_h_momentums[pid].Write('h_{}'.format(pid))
    h_total.Write('h_total')
    tf_out.Close()
    print('tf_out.Close()')


def plot_momentum_collimator_up():
    tf = TFile('data/beamline.py.radiation.collimator.in.pi+_85gev.15m.hist.root')
    for key in tf.GetListOfKeys():
        hist_name = key.GetName()
        hist = tf.Get(hist_name)

        particle_name = None
        if hist_name == 'h_total':
            particle_name = 'all'
        else:
            pid = int(hist_name.split('_')[1])
            particle_name = PDG.GetParticle(pid).GetName()

        if particle_name != 'all':
            continue

        if particle_name == 'antineutron':
            hist.Rebin(100)
        if particle_name == 'neutron':
            hist.Rebin(5)
        if particle_name == 'antiproton':
            hist.Rebin(100)
        if particle_name == 'proton':
            hist.Rebin(20)
        if particle_name == 'K+':
            hist.Rebin(50)
        if particle_name == 'K-':
            hist.Rebin(100)
        if particle_name == 'pi+':
            hist.Rebin(20)
            hist.GetXaxis().SetLimits(1, 1e5)
        if particle_name == 'pi-':
            hist.Rebin(20)
        if particle_name == 'mu+':
            hist.Rebin(50)
            hist.GetXaxis().SetLimits(1, 1e5)
        if particle_name == 'mu-':
            hist.Rebin(50)
            hist.GetXaxis().SetLimits(1, 1e5)

        c1 = TCanvas('c1', 'c1', 800, 600)
        set_margin()
        gPad.SetLogx()
        gPad.SetLogy()
        gPad.SetGrid()
        set_h1_style(hist)
        hist.Draw()
        hist.GetXaxis().SetTitle('Momentum (MeV)')
        hist.GetYaxis().SetTitle('Particle Count')

        c1.Update()
        hist.SetName(particle_name)
        draw_statbox(hist, x1=0.7, x2=0.98, y1=0.72, y2=0.94)

        c1.Update()
        c1.SaveAs('{}/plot_momentum_collimator_up.{}.pdf'.format(FIGURE_DIR, particle_name))
        input('Press any key to continue.')


def print_momentum_collimator_up():
    tf = TFile('data/beamline.py.radiation.collimator.in.pi+_85gev.15m.hist.root')

    particle_name_infos = {}
    for key in tf.GetListOfKeys():
        hist_name = key.GetName()
        if hist_name == 'h_total':
            continue
        pid = int(hist_name.split('_')[1])
        particle_name = PDG.GetParticle(pid).GetName()
        hist = tf.Get(hist_name)

        if particle_name not in particle_name_infos:
            particle_name_infos[particle_name] = {
                'mean': hist.GetMean(),
                'count': hist.GetEntries()
            }

    pprint(particle_name_infos)

    particle_names = sorted(particle_name_infos.keys(), key=lambda x: PDG.GetParticle(x).Mass())
    scaling_factor = 4.68e9 / 15.e6

    with open ('/Users/juntinghuang/beamer/20180912_testbeam_radiation_collimator/print_momentum_collimator_up.csv', 'w') as f_csv:
        f_csv.write('particle name, count per 1.5e7 particles on target, count per 4.68e9 particles on target, mean momentum (MeV)\n')

        count_all = 0.
        avg_momentum = 0.
        for particle_name in particle_names:
            info = particle_name_infos[particle_name]
            print('{} & \SI{{{:.1E}}}{{}} & \SI{{{:.1E}}}{{}} & {:.0f} \\\\'.format(particle_name, info['count'], info['count'] * scaling_factor, info['mean']))
            f_csv.write('{}, {:.3E}, {:.3E}, {:.3f}\n'.format(particle_name, info['count'], info['count'] * scaling_factor, info['mean']))
            count_all += info['count']
            avg_momentum += info['mean'] * info['count']

        avg_momentum /= count_all
        print('{} & \SI{{{:.1E}}}{{}} & \SI{{{:.1E}}}{{}} & {:.0f} \\\\'.format('all', count_all, count_all * scaling_factor, avg_momentum))
        f_csv.write('{}, {:.3E}, {:.3E}, {:.3f}\n'.format('all', count_all, count_all * scaling_factor, avg_momentum))

    with open('/Users/juntinghuang/beamer/20180912_testbeam_radiation_collimator/print_momentum_collimator_up.tex', 'w') as f_tex:
        for particle_name in particle_names:
            f_tex.write('')
            f_tex.write('\\begin{frame}\n')
            f_tex.write('  \\frametitle{{{}}}\n'.format(particle_name))
            f_tex.write('  \\begin{figure}\n')
            f_tex.write('    \\includegraphics[width = 0.8\\textwidth]{{figures/{{plot_momentum_collimator_up.{}}}.pdf}}\n'.format(particle_name))
            f_tex.write('    \\caption{{Momentum distribution for {} based on 15 million secondary beam particles on target.}}\n'.format(particle_name))
            f_tex.write('  \\end{figure}\n')
            f_tex.write('\\end{frame}\n\n')
            f_tex.write('% .........................................................\n\n')

            # print('\\begin{frame}')
            # print('  \\frametitle{{{}}}'.format(particle_name))
            # print('  \\begin{figure}')
            # print('    \\includegraphics[width = 0.8\\textwidth]{{figures/{{plot_momentum_collimator_up.{}}}.pdf}}'.format(particle_name))
            # print('    \\caption{{Momentum distribution for {} based on 10M secondary particle on target.}}'.format(particle_name))
            # print('  \\end{figure}')
            # print('\\end{frame}')
            # break


def plot_p_vs_angle_16_degree():
    field_length = 42. * INCH_TO_METER # m
    b_fields = [1.8, 1.35, 0.9, 0.45]
    colors = [kBlue + 2, kGreen + 2, kRed + 2, kBlack]
    degrees = np.arange(0.1, 32., 0.1)

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
    lg1 = TLegend(0.31, 0.56, 0.54, 0.85)
    set_legend_style(lg1)
    gPad.SetGrid()

    grs[0].Draw('AL')
    grs[0].GetXaxis().SetRangeUser(0., 25.)
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

    tl = TLine(16, 0, 16, 15)
    tl.SetLineWidth(3)
    tl.SetLineColor(kMagenta + 1)
    tl.SetLineStyle(10)
    tl.Draw()
    lg1.AddEntry(tl, '\\theta = 16^{\circ}', 'l')

    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_p_vs_angle_16_degree.pdf'.format(FIGURE_DIR))
    c1.SaveAs('{}/plot_p_vs_angle_16_degree.png'.format(FIGURE_DIR))
    input('Press any key to continue.')


def save_trigger_rate():
    tfile = TFile('{}/beamline.py.in.job_901_1800.9m.b_-0.9T.pi+_64gev.root'.format(DATA_DIR))

    pid_momentums = {}
    particles = []
    noise_particles = []
    charged_particles = [
        'e-', 'e+',
        'mu-', 'mu+',
        'pi-', 'pi+',
        'K-', 'K+',
        'antiproton', 'proton'
    ]

    h_present = TH1D('h_present', 'h_present', 8, 0, 8)
    h_pass = TH1D('h_pass', 'h_pass', 3, 0, 3)

    h_present.GetXaxis().CanExtend()
    h_pass.GetXaxis().CanExtend()

    detector_names = ['TOF US', 'WC 1', 'WC 2', 'WC 3', 'WC 4', 'TOF DS', 'Cherenkov', 'NOvA']
    pass_names = ['4 WC + 0 TOF', '3 WC + 2 TOF', 'All']

    keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
    for key in keys:
        print('key = {}'.format(key))
        track_count = 0
        for track in tfile.Get(key):
            track_count += 1
            if track_count % 100000 == 0:
                print('track_count = {}'.format(track_count))

            pid = int(track.PDGidstart_line)
            name = PDG.GetParticle(pid).GetName()
            if name not in charged_particles:
                continue

            pass_all = track.TrackPresentstart_line and \
                       track.TrackPresenttof_us and \
                       track.TrackPresentwire_chamber_1_detector and \
                       track.TrackPresentwire_chamber_2_detector and \
                       track.TrackPresentwire_chamber_3_detector and \
                       track.TrackPresentwire_chamber_4_detector and \
                       track.TrackPresenttof_ds and \
                       track.TrackPresentcherenkov and \
                       track.TrackPresentnova

            pass_4_wc_0_tof = track.TrackPresentstart_line and \
                              track.TrackPresentwire_chamber_1_detector and \
                              track.TrackPresentwire_chamber_2_detector and \
                              track.TrackPresentwire_chamber_3_detector and \
                              track.TrackPresentwire_chamber_4_detector

            pass_3_wc_2_tof = track.TrackPresentstart_line and \
                              track.TrackPresenttof_us and \
                              track.TrackPresentwire_chamber_1_detector and \
                              track.TrackPresentwire_chamber_2_detector and \
                              track.TrackPresentwire_chamber_3_detector and \
                              track.TrackPresentwire_chamber_4_detector and \
                              track.TrackPresenttof_ds

            track_presents = [
                track.TrackPresenttof_us,
                track.TrackPresentwire_chamber_1_detector,
                track.TrackPresentwire_chamber_2_detector,
                track.TrackPresentwire_chamber_3_detector,
                track.TrackPresentwire_chamber_4_detector,
                track.TrackPresenttof_ds,
                track.TrackPresentcherenkov,
                track.TrackPresentnova
            ]
            for i, track_present in enumerate(track_presents):
                if track_present:
                    h_present.Fill(detector_names[i], 1)

            track_passes = [pass_4_wc_0_tof, pass_3_wc_2_tof, pass_all]
            for i, track_pass in enumerate(track_passes):
                if track_pass:
                    h_pass.Fill(pass_names[i], 1)

            # if track_count % 1000 == 0:
            #     break
        # break

    tfile_out = TFile('{}/save_trigger_rate.root'.format(DATA_DIR), 'RECREATE')
    h_present.Write('h_present')
    h_pass.Write('h_pass')
    tfile_out.Close()


def plot_trigger_rate(**kwargs):
    plot_rate = kwargs.get('plot_rate', False)

    tfile = TFile('{}/save_trigger_rate.root'.format(DATA_DIR))
    h_present = tfile.Get('h_present')
    h_pass = tfile.Get('h_pass')

    h_present.Scale(1. / 9.)
    h_pass.Scale(1. / 9.)
    if plot_rate:
        h_present.Scale(1. / 4.2)
        h_pass.Scale(1. / 4.2)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    gPad.SetLogy()
    set_h1_style(h_present)
    h_present.LabelsDeflate('X')
    h_present.Draw('hist')
    h_present.GetYaxis().SetTitle('Particle Count per 1M Beam Particles')
    if plot_rate:
        h_present.GetYaxis().SetTitle('Particle Rate (Hz)')
    h_present.GetYaxis().SetTitleOffset(1.6)
    c1.Update()
    c1.SaveAs('{}/plot_trigger_rate.present.plot_rate_{}.pdf'.format(FIGURE_DIR, plot_rate))

    c2 = TCanvas('c2', 'c2', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_h1_style(h_pass)
    h_pass.GetYaxis().SetTitle('Coincidence Count per 1M Beam Particles')
    if plot_rate:
        h_pass.GetYaxis().SetTitle('Coincidence Rate (Hz)')
    h_pass.LabelsDeflate('X')
    h_pass.Draw('hist')
    c2.Update()
    c2.SaveAs('{}/plot_trigger_rate.pass.plot_rate_{}.pdf'.format(FIGURE_DIR, plot_rate))
    input('Press any key to continue.')


def compute_minimum_kinetic_energy():
    pids = [11, 13, -211, -321, 2212]
    # momentum = 200
    # momentum = 100
    # momentum = 3000
    momentum = 64000
    for pid in pids:
        mass = PDG.GetParticle(pid).Mass() * 1.e3
        name = PDG.GetParticle(pid).GetName()
        gamma = (momentum**2 / mass**2 + 1)**0.5
        kinetic_energy = (gamma - 1.) * mass
        kinetic_energy_2 = ((gamma - 1) / (gamma + 1))**0.5 * momentum

        # print('{}: gamma = {}, kinetic_energy = {}, kinetic_energy_2 = {}'.format(name, gamma, kinetic_energy, kinetic_energy_2))
        print('{}: gamma = {:.3f}, kinetic_energy = {:.1f}, kinetic_energy_2 = {:.1f}, E = {}'.format(name, gamma, kinetic_energy, kinetic_energy_2, gamma * mass))


def plot_b_field():
    with open('{}/SDG_Field_Map/SDG_Field_Map_082718a_1200A.txt'.format(DATA_DIR)) as f_txt:
        rows = csv.reader(f_txt, delimiter='\t')
        for i in range(4):
            next(rows)

        position_xs = set({})
        position_ys = set({})
        position_zs = set({})
        position_b_ys = {}
        for row in rows:
            position_x = float(row[0])
            position_y = float(row[1])
            position_z = float(row[2])
            b_x = float(row[3])
            b_y = float(row[4])
            b_z = float(row[5])

            position_xs.add(position_x)
            position_ys.add(position_y)
            position_zs.add(position_z)
            position_b_ys[(position_x, position_y, position_z)] = b_y

        position_xs = sorted(list(position_xs))
        position_ys = sorted(list(position_ys))
        position_zs = sorted(list(position_zs))
        max_position_x = max(position_xs)
        max_position_y = max(position_ys)
        max_position_z = max(position_zs)
        position_x_count = len(position_xs)
        position_y_count = len(position_ys)
        position_z_count = len(position_zs)
        position_step_x = max_position_x / position_x_count
        position_step_y = max_position_y / position_y_count
        position_step_z = max_position_z / position_z_count

        print('max_position_x = {}'.format(max_position_x))
        print('max_position_y = {}'.format(max_position_y))
        print('max_position_z = {}'.format(max_position_z))
        print('position_x_count = {}'.format(position_x_count))
        print('position_y_count = {}'.format(position_y_count))
        print('position_z_count = {}'.format(position_z_count))
        print('position_x_count * position_y_count * position_z_count = {}'.format(position_x_count * position_y_count * position_z_count))
        print('position_step_x = {}'.format(position_step_x))
        print('position_step_y = {}'.format(position_step_y))
        print('position_step_z = {}'.format(position_step_z))

        h_y = TH2D('h_y', 'h_y',
                   position_z_count, -position_step_z / 2., max_position_z + position_step_z / 2.,
                   position_x_count, -position_step_x / 2., max_position_x + position_step_x / 2.)
        for position_x in position_xs:
            for position_z in position_zs:
                h_y.Fill(position_z, position_x, abs(position_b_ys[(position_x, 0., position_z)]))

        h_z = TH2D('h_z', 'h_z',
                   position_x_count, -position_step_x / 2., max_position_x + position_step_x / 2.,
                   position_y_count, -position_step_y / 2., max_position_y + position_step_y / 2.)
        for position_x in position_xs:
            for position_y in position_ys:
                h_z.Fill(position_x, position_y, abs(position_b_ys[(position_x, position_y, 0.)]))

        h_x = TH2D('h_x', 'h_x',
                   position_z_count, -position_step_z / 2., max_position_z + position_step_z / 2.,
                   position_y_count, -position_step_y / 2., max_position_y + position_step_y / 2.)
        for position_y in position_ys:
            for position_z in position_zs:
                h_x.Fill(position_z, position_y, abs(position_b_ys[(0., position_y, position_z)]))

        c1 = TCanvas('c1', 'c1', 800, 600)
        set_margin()
        set_h2_color_style()
        set_h2_style(h_y)
        h_y.Draw('colz')
        h_y.GetXaxis().SetTitle('Z (m)')
        h_y.GetYaxis().SetTitle('X (m)')
        h_y.GetZaxis().SetTitle('B_{y} (T)')
        h_y.GetXaxis().SetTitleOffset(1.2)
        h_y.GetYaxis().SetTitleOffset(1.4)
        c1.Update()
        c1.SaveAs('{}/plot_b_field.h_y.pdf'.format(FIGURE_DIR))
        input('Press any key to continue.')

        h_y.GetXaxis().SetRangeUser(0, 0.5334)
        # h_y.GetXaxis().SetRangeUser(0, 0.5)
        c1.SaveAs('{}/plot_b_field.h_y.zoom.pdf'.format(FIGURE_DIR))
        input('Press any key to continue.')

        c2 = TCanvas('c2', 'c2', 800, 600)
        set_margin()
        set_h2_color_style()
        set_h2_style(h_z)
        h_z.Draw('colz')
        h_z.GetXaxis().SetTitle('X (m)')
        h_z.GetYaxis().SetTitle('Y (m)')
        h_z.GetZaxis().SetTitle('B_{y} (T)')
        h_z.GetXaxis().SetTitleOffset(1.2)
        h_z.GetYaxis().SetTitleOffset(1.4)
        c2.Update()
        c2.SaveAs('{}/plot_b_field.h_z.pdf'.format(FIGURE_DIR))
        input('Press any key to continue.')

        c3 = TCanvas('c3', 'c3', 800, 600)
        set_margin()
        set_h2_color_style()
        set_h2_style(h_x)
        h_x.Draw('colz')
        h_x.GetXaxis().SetTitle('Z (m)')
        h_x.GetYaxis().SetTitle('Y (m)')
        h_x.GetZaxis().SetTitle('B_{y} (T)')
        h_x.GetXaxis().SetTitleOffset(1.2)
        h_x.GetYaxis().SetTitleOffset(1.4)
        c3.Update()
        c3.SaveAs('{}/plot_b_field.h_x.pdf'.format(FIGURE_DIR))
        input('Press any key to continue.')

        h_x.GetXaxis().SetRangeUser(0, 0.5334)
        # h_x.GetXaxis().SetRangeUser(0, 0.5)
        c3.SaveAs('{}/plot_b_field.h_x.zoom.pdf'.format(FIGURE_DIR))
        input('Press any key to continue.')


def split_rows(filename, split_size):
    items = []
    with open(filename) as f_in:
        for row in csv.reader(f_in, delimiter=' '):
            # items = row
            # break
            items.extend(row)
    split_items = split_list(items, split_size)
    filename_base = os.path.splitext(filename)[0]
    for i, split_items in enumerate(split_items):
        filename_out = '{}.{}.{}.txt'.format(filename_base, len(split_items), i)
        with open(filename_out, 'w') as f_out:
            f_out.write(' '.join(split_items))


def plot_particle_count_vs_b_field(**kwargs):
    y_axis_title = kwargs.get('y_axis_title', 'Good Particles per 1M Beam Particle')
    scaling_factor = kwargs.get('scaling_factor', None)
    filenames = kwargs.get('filenames', ['g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
                                         'g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root',
                                         'g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
                                         'g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root.hist.root'])
    b_fields = kwargs.get('b_fields', [-0.45, -0.9, -1.35, -1.8])
    pids = kwargs.get('pids', [-11, -13, 211, 321, 2212])
    suffix = kwargs.get('suffix', '')
    canvas_width = kwargs.get('canvas_width', 800)
    canvas_height = kwargs.get('canvas_height', 600)

    b_field_sign = np.sign(b_fields[0])
    b_fields = list(map(abs, b_fields))

    colors = [
        kBlack,
        kRed,
        kBlue,
        kMagenta + 1,
        kGreen + 2
    ]

    tfs = []
    for filename in filenames:
        tfs.append(TFile('{}/{}'.format(DATA_DIR, filename)))

    grs = []
    for pid in pids:
        particle_counts = []
        for tf in tfs:
            hist = tf.Get('h_{}'.format(pid))
            particle_count = 0.
            try:
                particle_count = float(hist.Integral())
            except AttributeError as err:
                print('pid = {}'.format(pid))
                print(err)

            if scaling_factor:
                particle_count *= scaling_factor
            particle_counts.append(particle_count)
        gr = TGraph(len(b_fields), np.array(b_fields), np.array(particle_counts))
        set_graph_style(gr)
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', canvas_width, canvas_height)
    set_margin()
    gPad.SetLogy()
    gPad.SetGrid()

    lg1 = TLegend(0.46, 0.18, 0.76, 0.42)
    set_legend_style(lg1)

    for i, gr in enumerate(grs):
        gr.SetLineColor(colors[i])
        gr.SetMarkerColor(colors[i])
        if i == 0:
            gr.Draw('APL')
            gr.GetYaxis().SetRangeUser(0.001, 50.)
            if scaling_factor:
                gr.GetYaxis().SetRangeUser(0.001 * scaling_factor, 50. * scaling_factor)
            gr.GetXaxis().SetTitle('B Field (T)')
            if b_field_sign < 0:
                gr.GetXaxis().SetTitle('B Field (#minusT)')
            gr.GetYaxis().SetTitle(y_axis_title)
            if canvas_height == 800:
                gr.GetYaxis().SetTitleOffset(2.)
        else:
            gr.Draw('PL')
        lg1.AddEntry(gr, PDG.GetParticle(pids[i]).GetName(), 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_particle_count_vs_b_field.scaling_factor_{}{}.pdf'.format(FIGURE_DIR, scaling_factor, suffix))
    input('Press any key to continue.')


def print_particle_count_vs_b_field(**kwargs):
    filenames = kwargs.get('filenames', ['g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
                                         'g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root',
                                         'g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
                                         'g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root.hist.root'])
    b_fields = kwargs.get('b_fields', [-0.45, -0.9, -1.35, -1.8])
    pids = kwargs.get('pids', [211, 2212, -11, -13, 321])

    b_field_strs = [str(b_field) for b_field in b_fields]
    pids = sorted(pids, key=abs)
    tfs = []
    for filename in filenames:
        tfs.append(TFile('{}/{}'.format(DATA_DIR, filename)))

    pid_infos = {}
    for pid in pids:
        pid_infos[pid] = {}
        for i, tf in enumerate(tfs):
            hist = tf.Get('h_{}'.format(pid))
            count = 0.
            mean = 0.
            rms = 0.
            try:
                count = hist.Integral()
                mean = hist.GetMean()
                rms = hist.GetRMS()
            except AttributeError as err:
                print('pid = {}'.format(pid))
                print(err)

            b_field_str = str(b_field_strs[i])
            pid_infos[pid][b_field_str] = {}
            pid_infos[pid][b_field_str]['count'] = count
            pid_infos[pid][b_field_str]['count_month'] = count * 60 * 24 * 30
            pid_infos[pid][b_field_str]['mean'] = mean
            pid_infos[pid][b_field_str]['rms'] = rms

    info_names = ['count', 'count_month', 'mean', 'rms']
    for i, pid in enumerate(pids):
        particle_name = PDG.GetParticle(pid).GetName()
        print('\hline')
        print('\hline')
        for i, b_field_str in enumerate(b_field_strs):
            row = ''
            if i == 0:
                row += '\\multirow{{{}}}{{*}}{{{}}}'.format(len(b_field_strs), particle_name)
            row += '& ' + b_field_str
            for info_name in info_names:
                row += ' & \\SI{{{:.2E}}}{{}}'.format(pid_infos[pid][b_field_str][info_name])
            row += ' \\\\'
            print(row)


def plot_beamline_sim_global_timing():
    tf = TFile('{}/g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root'.format(DATA_DIR))

    h1 = TH1D('h1', 'h1', 48000, 0, 48000)
    for event in tf.Get('tree'):
        if event.is_noise == 1:
            continue
        h1.Fill(event.t)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)

    h1.GetXaxis().SetRangeUser(0, 600)
    h1.GetXaxis().SetTitle('Time (s)')
    h1.GetYaxis().SetTitle('Good Particle Count per Second')
    h1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_beamline_sim_global_timing.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_beamline_sim_spill_timing():
    beam_particle_count_per_spill = 1e6

    tf = TFile('{}/g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root'.format(DATA_DIR))
    h1 = TH1D('h1', 'h1', 50, 0, 5)

    spill_number_max = 0
    for event in tf.Get('tree'):
        if event.is_noise == 1:
            continue
        spill_number = int(event.event_id / beam_particle_count_per_spill)
        if spill_number > spill_number_max:
            spill_number_max = spill_number
        h1.Fill(event.t - 60. - spill_number * 60.)

    print('spill_number_max = {}'.format(spill_number_max))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)

    h1.GetXaxis().SetRangeUser(0, 600)
    h1.GetXaxis().SetTitle('Time (s)')
    h1.GetYaxis().SetTitle('Good Particle Count')
    h1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_beamline_sim_spill_timing.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_detsim_momentum(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_electron = tf.Get('testbeamana/fMcMomentumElectron')
    h_muon = tf.Get('testbeamana/fMcMomentumMuon')
    h_pion = tf.Get('testbeamana/fMcMomentumPion')
    h_kaon = tf.Get('testbeamana/fMcMomentumKaon')
    h_proton = tf.Get('testbeamana/fMcMomentumProton')

    h_particles = [h_electron, h_muon, h_pion, h_kaon, h_proton]
    colors = [
        kBlack,
        kRed,
        kBlue,
        kMagenta + 1,
        kGreen + 2
    ]
    legend_names = ['e+', 'mu+', 'pi+', 'K+', 'proton']

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()

    y_max = 0.
    for h_particle in h_particles:
        h_particle.Scale(1. / 800.)
        h_particle.Rebin(2)
        if h_particle.GetMaximum() > y_max:
            y_max = h_particle.GetMaximum()
    y_max *= 1.2

    lg1 = TLegend(0.64, 0.6, 0.80, 0.84)
    set_legend_style(lg1)

    for i, h_particle in enumerate(h_particles):
        set_h1_style(h_particle)
        h_particle.SetLineColor(colors[i])
        if i == 0:
            h_particle.GetXaxis().SetTitle('Momentum (GeV)')
            h_particle.GetYaxis().SetTitle('Particle Count per 1M Beam Particle')
            h_particle.Draw('hist')
            h_particle.GetXaxis().SetRangeUser(1., 2.5)
            h_particle.GetYaxis().SetRangeUser(1e-3, y_max)
        else:
            h_particle.Draw('hist,sames')

        lg1.AddEntry(h_particle, legend_names[i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_detsim_momentum.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_detsim_fls_hit_gev(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h_electron = tf.Get('testbeamana/fFlsHitGevElectron')
    h_muon = tf.Get('testbeamana/fFlsHitGevMuon')
    h_pion = tf.Get('testbeamana/fFlsHitGevPion')
    h_kaon = tf.Get('testbeamana/fFlsHitGevKaon')
    h_proton = tf.Get('testbeamana/fFlsHitGevProton')

    h_particles = [h_electron, h_muon, h_pion, h_kaon, h_proton]
    colors = [
        kBlack,
        kRed,
        kBlue,
        kMagenta + 1,
        kGreen + 2
    ]
    legend_names = ['e+', 'mu+', 'pi+', 'K+', 'proton']

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    y_max = 0.
    for h_particle in [h_pion, h_proton]:
        h_particle.Scale(1. / h_particle.Integral())
        if h_particle.GetMaximum() > y_max:
            y_max = h_particle.GetMaximum()
    y_max *= 1.1

    lg1 = TLegend(0.64, 0.6, 0.80, 0.84)
    set_legend_style(lg1)

    for i, h_particle in enumerate([h_pion, h_proton]):
        set_h1_style(h_particle)
        h_particle.SetLineColor(colors[i])

        if i == 0:
            h_particle.GetXaxis().SetRangeUser(0, 2)
            h_particle.GetXaxis().SetTitle('Total Energy Deposit in Liquid Scintillator (GeV)')
            h_particle.GetYaxis().SetTitle('Event Count')
            h_particle.GetYaxis().SetRangeUser(0, y_max)
            h_particle.Draw('hist')
        else:
            h_particle.Draw('hist,sames')
        lg1.AddEntry(h_particle, ['pi+', 'proton'][i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_detsim_fls_hit_gev.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')

def print_figure_tex():
    figures = ['plot_saved_particle_momentum.g4bl.b_-0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-0.675T.proton.64000.root.job_1_20000.799.72m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_0.675T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_0.9T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_1.35T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root.pdf',
               'plot_saved_particle_momentum.g4bl.b_1.8T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root.pdf']

    figures = ['{' + figure.replace('.pdf', '}.pdf') for figure in figures]
    # print('figures = {}'.format(figures))

    b_fields = [-0.225, -0.45, -0.675, -0.9, -1.125, -1.35, -1.575, -1.8, 0.225, 0.45, 0.675, 0.9, 1.125, 1.35, 1.575, 1.8]

    for i, b_field in enumerate(b_fields):
        if b_field < 0.:
            continue
        print('  \\begin{subfigure}[c]{0.49\\textwidth}')
        print('    \\centering')
        print('    \\includegraphics[width=0.9\\textwidth]{{figures/{}}}'.format(figures[i]))
        # print('    \\caption{{$B = \SI{{{:.3f}}}{{T}}$}}'.format(b_field))
        print('  \\end{subfigure}')


def plot_noise_particle_root(filename, **kwargs):
    log_y = kwargs.get('log_y', False)
    show_boundary = kwargs.get('show_boundary', False)

    width = 2606.2 / 10.       # cm
    half_width = width / 2.    # cm
    x0 = -1354.4 / 10.         # cm
    y0 = 0.
    margin = 20.
    pid_y_x_hists = {}
    pid_momentum_x_hists = {}

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    tree = tf.Get('tree')
    particle_count_total = tree.GetEntries()
    particle_count = 0

    for particle in tf.Get('tree'):
        particle_count += 1
        if particle_count % 1e6 == 0:
            print('particle_count = {} / {} ({:.1f}%)'.format(particle_count, particle_count_total, particle_count / particle_count_total * 100.))

        if not particle.is_noise:
            continue
        pid = int(particle.pdg_id)
        px = particle.px
        py = particle.py
        pz = particle.pz
        momentum = (px**2 + py**2 + pz**2)**0.5
        x = particle.x
        y = particle.y
        z = particle.z
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
        c1.SaveAs('{}/plot_noise_particle_root.y_x.{}.pid_{}.pdf'.format(FIGURE_DIR, filename, PDG.GetParticle(pid).GetName()))

    # for pid, h1 in pid_momentum_x_hists.items():
    #     set_h2_style(h1)
    #     h1.Draw('colz')
    #     h1.GetXaxis().SetTitle('X (cm)')
    #     h1.GetYaxis().SetTitle('Momentum (MeV)')
    #     h1.GetXaxis().SetTitleOffset(1.8)
    #     h1.GetYaxis().SetTitleOffset(2.)
    #     c1.Update()
    #     c1.SaveAs('{}/plot_noise_particle_root.momentum_x.{}.pid_{}.pdf'.format(FIGURE_DIR, filename, PDG.GetParticle(pid).GetName()))
    # input('Press any key to continue.')


def plot_det_sim_particle_count_per_event(filename, **kwargs):
    x_max = kwargs.get('x_max', None)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h1 = tf.Get('h_particle_count_per_event')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()
    gPad.SetLogx()
    set_h1_style(h1)
    h1.GetXaxis().SetTitle('Particle Count per Event')
    h1.GetYaxis().SetTitle('Event Count')
    h1.GetYaxis().SetMaxDigits(3)
    if x_max:
        h1.GetXaxis().SetRangeUser(0.5, x_max)
    h1.Draw()

    c1.Update()
    draw_statbox(h1)

    c1.Update()
    c1.SaveAs('{}/save_to_txt.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def compare_det_sim_particle_count_per_event(filenames, hist_names, colors, suffix):
    h1s = []
    for filename in filenames:
        tf1 = TFile('{}/{}'.format(DATA_DIR, filename))
        h1 = tf1.Get('h_particle_count_per_event')
        h1.SetDirectory(0)
        h1s.append(h1)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()
    gPad.SetLogx()

    for i, h1 in enumerate(h1s):
        set_h1_style(h1)
        h1.SetName(hist_names[i])
        h1.SetLineColor(colors[i])
        if i == 0:
            h1.GetXaxis().SetTitle('Particle Count per Event')
            h1.GetYaxis().SetTitle('Event Count')
            h1.GetYaxis().SetMaxDigits(3)
            h1.GetXaxis().SetRangeUser(0.1, 50)
            h1.GetYaxis().SetRangeUser(0.1, 1e4)
            h1.Draw()
        else:
            print('i = {}'.format(i))
            h1.Draw('sames')
        print('h1.GetName() = {}'.format(h1.GetName()))

    c1.Update()
    if len(filenames) == 2:
        draw_statboxes([h1s[1], h1s[0]],
                       width=0.18, height=0.15,
                       corner_x=0.2, corner_y=0.3,
                       gap_y=0.02)
    elif len(filenames) == 3:
        draw_statboxes([h1s[2], h1s[1], h1s[0]],
                       width=0.18, height=0.15,
                       corner_x=0.2, corner_y=0.2,
                       gap_y=0.02)
    elif len(filenames) == 4:
        draw_statboxes([h1s[3], h1s[2], h1s[1], h1s[0]],
                       width=0.15, height=0.12,
                       corner_x=0.2, corner_y=0.18,
                       gap_y=0.02)

    c1.Update()
    c1.SaveAs('{}/compare_det_sim_particle_count_per_event.{}.pdf'.format(FIGURE_DIR, suffix))
    input('Press any key to continue.')


def get_saved_particle_count(filename):
    pid_particle_counts = {}
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    for key in tf.GetListOfKeys():
        hist_name = key.GetName()
        if hist_name == 'h_all':
            continue
        pid = int(hist_name.split('_')[1])
        pid_particle_counts[pid] = tf.Get(hist_name).Integral()

    return pid_particle_counts


def print_saved_particle_count(**kwargs):
    noise_particle = kwargs.get('noise_particle', True)
    filenames = kwargs.get('filenames', [
        'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root.hist.root',
        'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root.hist.root',
        'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_3.root.hist.root'
    ])

    config_pid_counts = []
    for filename in filenames:
        config_pid_counts.append(get_saved_particle_count(filename))

    pids = []
    for pid in config_pid_counts[0].keys():
        if not noise_particle and (PDG.GetParticle(pid).Charge() == 0 or (PDG.GetParticle(pid).Charge() < 0. and b_field < 0.) or (PDG.GetParticle(pid).Charge() > 0. and b_field > 0.)):
            print('Wrong sign particles: pid = {}, count = {}, avg momentum = {}'.format(pid, pid_hists[pid].Integral(), pid_hists[pid].GetMean()))
            continue
        pids.append(pid)
    pids = sorted(pids, key=lambda x: (abs(x), np.sign(x)))

    for pid in pids:
        # row = '{} & {}'.format(str(pid), PDG.GetParticle(pid).GetName())
        row = PDG.GetParticle(pid).GetName()
        for i, config_pid_count in enumerate(config_pid_counts):
            row += ' & {:.1E}'.format(config_pid_count[pid])
            if i == len(config_pid_counts) - 1:
                row += ' & {:.0f}\%'.format((config_pid_count[pid] - config_pid_counts[0][pid]) / config_pid_counts[0][pid] * 100.)

        row += ' \\\\'
        print(row)
        # print('row = {}'.format(row))
        # break


def plot_particle_timing_detector_event():
    tf = TFile('data/text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.no_shielding_2.root.root')
    h_timing = tf.Get('h_timing')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()

    set_h1_style(h_timing)
    h_timing.SetLineColor(kRed)
    h_timing.GetXaxis().SetTitle('Time (ns)')
    h_timing.GetYaxis().SetTitle('Number of Particles')
    h_timing.Draw()
    c1.Update()
    draw_statbox(h_timing)

    c1.Update()
    c1.SaveAs('{}/plot_particle_timing_detector_event.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def print_momentum_resolution():
    b_field = 0.9                        # tesla
    b_field_length = 42. * INCH_TO_METER # m
    theta = 16. * DEGREE_TO_RADIAN       # radian

    medium_length = 48. * INCH_TO_METER # m
    radiation_length_helium = 5671.     # m
    radiation_length_air = 304.         # m
    radiation_length = radiation_length_air
    # radiation_length = radiation_length_helium

    particle_name = 'proton'
    # particle_name = 'pi+'
    # particle_name = 'e+'
    mass = PDG.GetParticle(particle_name).Mass() * 1000. # MeV
    momentum = b_field * b_field_length * SPEED_OF_LIGHT / theta * 1.e-6 # MeV
    beta = momentum / (momentum**2 + mass**2)**0.5
    charge_number = 1.

    dtheta = 13.6 / (beta * momentum) * charge_number * (medium_length / radiation_length)**0.5 * (1. + 0.038 * log(medium_length / radiation_length * charge_number**2 / beta**2))
    dp = b_field * b_field_length * SPEED_OF_LIGHT / theta**2 * dtheta * 1.e-6

    print('mass = {}'.format(mass))
    print('momentum = {}'.format(momentum))
    print('medium_length = {}'.format(medium_length))
    print('dp = {}'.format(dp))
    print('dp / momentum = {}'.format(dp / momentum))


def get_momentum_resolution(b_field, particle_name, radiation_length):
    b_field_length = 42. * INCH_TO_METER # m
    theta = 16. * DEGREE_TO_RADIAN       # radian
    medium_length = 48. * INCH_TO_METER  # m

    mass = PDG.GetParticle(particle_name).Mass() * 1000. # MeV
    momentum = b_field * b_field_length * SPEED_OF_LIGHT / theta * 1.e-6 # MeV
    beta = momentum / (momentum**2 + mass**2)**0.5
    charge_number = 1.

    dtheta = 13.6 / (beta * momentum) * charge_number * (medium_length / radiation_length)**0.5 * (1. + 0.038 * log(medium_length / radiation_length * charge_number**2 / beta**2))
    dp = b_field * b_field_length * SPEED_OF_LIGHT / theta**2 * dtheta * 1.e-6

    return dp / momentum


def plot_momentum_resolution(medium):
    radiation_length_helium = 5671.     # m
    radiation_length_air = 304.         # m
    radiation_length_helium_pipe = get_radiation_length_helium_pipe()

    radiation_length = None
    if medium == 'air':
        radiation_length = radiation_length_air
    elif medium == 'helium':
        radiation_length = radiation_length_helium
    elif medium == 'helium_pipe':
        radiation_length = radiation_length_helium_pipe

    b_fields = np.arange(0.2, 2., 0.01)
    particle_names = ['proton', 'K+', 'pi+', 'mu+', 'e+']
    colors = [kBlack, kBlue, kRed, kGreen + 1, kMagenta + 1]

    grs = []
    for particle_name in particle_names:
        momentum_resolutions = []
        for b_field in b_fields:
            momentum_resolutions.append(get_momentum_resolution(b_field, particle_name, radiation_length) * 100.)
        gr = TGraph(len(b_fields), np.array(b_fields), np.array(momentum_resolutions))
        grs.append(gr)

    c1 = TCanvas('c1', 'c1', 600, 600)
    set_margin()
    gPad.SetGrid()

    lg1 = TLegend(0.55, 0.53, 0.95, 0.83)
    set_legend_style(lg1)

    for i, gr in enumerate(grs):
        set_graph_style(gr)
        gr.SetLineColor(colors[i])
        lg1.AddEntry(gr, particle_names[i], 'l')
        if i == 0:
            gr.Draw('AL')
            gr.GetXaxis().SetRangeUser(0.2, 2)
            gr.GetXaxis().SetTitle('B Field (T)')
            gr.GetYaxis().SetTitle('Momentum Resolution dp/p (%)')
            gr.GetYaxis().SetTitleOffset(1.4)
        gr.Draw('L, sames')

    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_momentum_resolution.{}.pdf'.format(FIGURE_DIR, medium))
    input('Press any key to continue.')


def get_radiation_length_helium_pipe():
    density_helium = 0.1664     # kg/m-3
    density_mylar = 1.4e3       # kg/m-3

    length_mylar = 152.4e-6     # m
    length_helium = 1.22        # m

    radiation_length_helium = 5671.     # m
    radiation_length_mylar = 0.2854     # m

    mass_fraction_helium = length_helium * density_helium / (length_helium * density_helium + length_mylar * density_mylar)
    mass_fraction_mylar = length_mylar * density_mylar / (length_helium * density_helium + length_mylar * density_mylar)

    length_fraction_helium = length_helium / (length_helium + length_mylar)
    length_fraction_mylar = length_mylar / (length_helium + length_mylar)

    # radiation_length_helium_pipe = 1. / (mass_fraction_helium / radiation_length_helium + mass_fraction_mylar / radiation_length_mylar)
    radiation_length_helium_pipe = 1. / (length_fraction_helium / radiation_length_helium + length_fraction_mylar / radiation_length_mylar)

    print('mass_fraction_helium = {}'.format(mass_fraction_helium))
    print('mass_fraction_mylar = {}'.format(mass_fraction_mylar))
    print('radiation_length_helium_pipe = {}'.format(radiation_length_helium_pipe))

    return radiation_length_helium_pipe


def plot_cherenkov_index_of_refaction_air():
    names = ['pi+', 'mu+', 'e+']
    colors = [kBlue + 1, kGreen + 2, kRed + 1]
    masses = list(map(lambda x: PDG.GetParticle(x).Mass(), names)) # GeV
    eta = 4.1e-4                  # atm-1, for CO2

    momentums = np.arange(0.01, 20, 0.01)
    rrefraction_indexs = []
    for i, mass in enumerate(masses):
        refraction_indexs = []
        for momentum in momentums:
            # pressure = 1. / eta * ((1 + (mass / momentum)**2)**0.5 - 1.)
            refraction_index = (1 + (mass / momentum)**2)**0.5
            refraction_indexs.append(refraction_index)
        rrefraction_indexs.append(refraction_indexs)

    grs = []
    for i in range(len(rrefraction_indexs)):
        gr = TGraph(len(momentums), np.array(momentums), np.array(rrefraction_indexs[i]))
        set_graph_style(gr)
        grs.append(gr)

    refraction_index_one_atm = eta * 1. + 1.;
    one_atm_refraction_indexs = [refraction_index_one_atm for i in range(len(momentums))]
    gr_one_atm = TGraph(len(momentums), np.array(momentums), np.array(one_atm_refraction_indexs))

    refraction_index_one_atm_air = 1.0003
    # refraction_index_one_atm_nitrogen = 1.0003
    refraction_index_one_atm_helium = 1.000035
    one_atm_air_refraction_indexs = [refraction_index_one_atm_air for i in range(len(momentums))]
    gr_one_atm_air = TGraph(len(momentums), np.array(momentums), np.array(one_atm_air_refraction_indexs))
    one_atm_helium_refraction_indexs = [refraction_index_one_atm_helium for i in range(len(momentums))]
    gr_one_atm_helium = TGraph(len(momentums), np.array(momentums), np.array(one_atm_helium_refraction_indexs))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    gPad.SetLeftMargin(0.2)
    # gPad.SetLogx()

    lg1 = TLegend(0.51, 0.56, 0.86, 0.86)
    set_legend_style(lg1)

    for i in range(len(names)):
        grs[i].SetLineColor(colors[i])
        lg1.AddEntry(grs[i], '{} threshold'.format(names[i]), 'l')

        if i == 0:
            grs[0].Draw('AL')
            grs[0].GetXaxis().SetRangeUser(0., 20)
            grs[0].GetYaxis().SetRangeUser(1., 1.0010)
            grs[0].GetYaxis().SetDecimals()
            grs[0].GetYaxis().SetTitleOffset(2)
            grs[0].GetYaxis().SetTitle('Index of Refraction')
            grs[0].GetXaxis().SetTitle('Momentum (GeV)')
        grs[i].Draw('sames,L')

    set_graph_style(gr_one_atm)
    gr_one_atm.SetLineStyle(9)
    gr_one_atm.SetLineColor(kBlue)
    gr_one_atm.SetLineWidth(2)
    gr_one_atm.Draw('L')
    lg1.AddEntry(gr_one_atm, 'CO_{2} at 1 atm ', 'l')

    set_graph_style(gr_one_atm_air)
    gr_one_atm_air.SetLineStyle(10)
    gr_one_atm_air.SetLineColor(kRed)
    gr_one_atm_air.SetLineWidth(2)
    gr_one_atm_air.Draw('L')
    lg1.AddEntry(gr_one_atm_air, 'air / N_{2} at 1 atm ', 'l')

    set_graph_style(gr_one_atm_helium)
    gr_one_atm_helium.SetLineStyle(3)
    gr_one_atm_helium.SetLineColor(kMagenta + 1)
    gr_one_atm_helium.SetLineWidth(2)
    gr_one_atm_helium.Draw('L')
    lg1.AddEntry(gr_one_atm_helium, 'helium at 1 atm ', 'l')

    # latex = TLatex()
    # latex.SetNDC()
    # latex.SetTextFont(43)
    # latex.SetTextSize(28)
    # latex.DrawLatex(0.5, 0.25, '1 atm')

    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_cherenkov_index_of_refaction_air.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_cerenkov_trigger_rate_vs_threshold():
    # thresholds = [100.1, 80.0, 50.1, 39.9, 35.0, 29.9]
    # start_times = [
    #     datetime(2019, 2, 27, 15, 44, 51),
    #     datetime(2019, 2, 27, 15, 58, 25),
    #     datetime(2019, 2, 27, 16, 5, 23),
    #     datetime(2019, 2, 27, 16, 43, 12),
    #     datetime(2019, 2, 27, 16, 33, 26),
    #     datetime(2019, 2, 27, 16, 23, 16),
    # ]
    # end_times = [
    #     datetime(2019, 2, 27, 15, 46, 28),
    #     datetime(2019, 2, 27, 16, 0, 47),
    #     datetime(2019, 2, 27, 16, 8, 34),
    #     datetime(2019, 2, 27, 16, 46, 9),
    #     datetime(2019, 2, 27, 16, 36, 10),
    #     datetime(2019, 2, 27, 16, 27, 44),
    # ]
    # trigger_counts = [
    #     43, 76, 181, 235, 332, 553
    # ]
    # durations = []              # minute
    # for i in range(len(start_times)):
    #     durations.append((end_times[i] - start_times[i]).total_seconds() / 60.)
    # avg_duration = sum(durations) / len(durations)
    # print('avg_duration = {}'.format(avg_duration))
    # trigger_counts = list(map(float, trigger_counts))
    # trigger_rates = []
    # for i, duration in enumerate(durations):
    #     trigger_rates.append(trigger_counts[i] / durations[i])

    fragment_duration = 40.         # s
    threshold_run_numbers = [
        (100.1, 2321),
        (80.0, 2322),
        (50.1, 2323),
        (39.9, 2327),
        (35.0, 2326),
        (29.9, 2325),
        # (30.0, 2368),
        # (35.0, 2369),
    ]
    threshold_run_numbers = sorted(threshold_run_numbers, key=lambda x: x[0])
    thresholds = []
    trigger_rates = []
    for threshold_run_number in threshold_run_numbers:
        threshold = threshold_run_number[0]
        thresholds.append(threshold)

        run_number = threshold_run_number[1]
        tf = TFile('{}/cerenkovana.cosmic_threshold_run_{}.root'.format(DATA_DIR, run_number))
        avg_event_count_per_fragment = tf.Get('cerenkovana/fAvgEventCountPerFragment').GetBinContent(1)
        trigger_rates.append(avg_event_count_per_fragment / fragment_duration)

        total_fragment_count = tf.Get('cerenkovana/fEventCountPerFragment').GetEntries()
        print('total_fragment_count = {}'.format(total_fragment_count))

    gr = TGraph(len(thresholds), np.array(thresholds), np.array(trigger_rates))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()

    set_graph_style(gr)
    gr.Draw('APL')
    gr.GetYaxis().SetRangeUser(0, 17)
    gr.GetXaxis().SetTitle('Threshold (mV)')
    gr.GetYaxis().SetTitle('Trigger Rate (Hz)')

    c1.Update()
    c1.SaveAs('{}/plot_cerenkov_trigger_rate_vs_threshold.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def get_graph_from_th1(h1, **kwarg):
    x_scale = kwarg.get('x_scale', 1.)

    adcs = []
    time_ticks = []
    for i in range(1, h1.GetNbinsX() + 1):
        adcs.append(h1.GetBinContent(i))
        time_ticks.append(float(i) * x_scale)
    return TGraph(len(time_ticks), np.array(time_ticks), np.array(adcs))


def plot_cerenkov_pulse(filename, **kwargs):
    gate_min = kwargs.get('gate_min', None)
    gate_max = kwargs.get('gate_max', None)
    spill = kwargs.get('spill', 1)
    event = kwargs.get('event', 0)
    channel = kwargs.get('channel', 2)
    trigger_channels = kwargs.get('trigger_channels', None)
    y_min = kwargs.get('y_min', None)
    y_max = kwargs.get('y_max', None)
    legend_ndcs = kwargs.get('legend_ndcs', [0.4, 0.22, 0.53, 0.45])

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    gr = get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, channel)))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    set_graph_style(gr)
    gr.Draw('AL')
    gr.GetXaxis().SetRangeUser(0, 1024)
    gr.GetXaxis().SetTitle('Time Tick (200 ps)')
    gr.GetYaxis().SetTitle('ADC')
    gr.GetYaxis().SetTitleOffset(1.5)
    if y_min is not None and y_max is not None:
        gr.GetYaxis().SetRangeUser(y_min, y_max)

    lg1 = TLegend(legend_ndcs[0], legend_ndcs[1], legend_ndcs[2], legend_ndcs[3])
    set_legend_style(lg1)
    lg1.AddEntry(gr, 'Cerenkov', 'l')
    lg1.Draw()

    if gate_min and gate_max:
        c1.Update()
        line_min = TLine(gate_min, gPad.GetUymin(), gate_min, gPad.GetUymax())
        line_max = TLine(gate_max, gPad.GetUymin(), gate_max, gPad.GetUymax())
        for line in [line_min, line_max]:
            line.Draw()
            line.SetLineWidth(2)
            line.SetLineStyle(2)
            line.SetLineColor(kBlue)
        lg1.AddEntry(line_min, 'gate', 'l')

    if trigger_channels:
        gr_trigger_1 = get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, trigger_channels[0])))
        gr_trigger_2 = get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, trigger_channels[1])))

        set_graph_style(gr_trigger_1)
        gr_trigger_1.SetLineColor(kRed)
        gr_trigger_1.Draw('L,sames')

        set_graph_style(gr_trigger_2)
        gr_trigger_2.SetLineColor(kBlue)
        gr_trigger_2.Draw('L,sames')

        lg1.AddEntry(gr_trigger_1, 'downstream trigger', 'l')
        lg1.AddEntry(gr_trigger_2, 'upstream trigger', 'l')

    c1.Update()
    c1.SaveAs('{}/plot_cerenkov_pulse.{}.spill_{}.event_{}.channel_{}{}.pdf'.format(FIGURE_DIR, filename, spill, event, channel, '.trigger' if trigger_channels else ''))
    input('Press any key to continue.')


def get_graph_min_max(gr):
    x_list = list(gr.GetX())
    y_list = list(gr.GetY())
    return min(x_list), max(x_list), min(y_list), max(y_list)


def print_integrated_charge_npe(filename, **kwargs):
    spill = kwargs.get('spill', 1)
    event = kwargs.get('event', 0)
    channel = kwargs.get('channel', 15)
    gate_min = kwargs.get('gate_min', 500)
    gate_max = kwargs.get('gate_max', 600)
    calibration_constant = kwargs.get('calibration_constant', None)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h1 = tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, channel))
    gr = get_graph_from_th1(h1)

    total_charge = 0.
    h_adc = TH1D('h_adc', 'h_adc', 4096, 0.5, 4096.5)
    for i in range(1, h1.GetNbinsX() + 1):
        h_adc.Fill(h1.GetBinContent(i))
        if i >= gate_min and i <= gate_max:
            total_charge += h1.GetBinContent(i)
    pedestal = h_adc.GetBinCenter(h_adc.GetMaximumBin())
    total_charge = pedestal * (gate_max - gate_min + 1) - total_charge
    print('spill = {}'.format(spill))
    print('event = {}'.format(event))
    print('channel = {}'.format(channel))
    print('pedestal = {}'.format(pedestal))
    print('total_charge = {} ADC*0.2ns'.format(total_charge))

    if calibration_constant:
        npe = total_charge * calibration_constant
        print('npe = {}'.format(npe))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gr.Draw('AL')
    c1.Update()
    line_min = TLine(gate_min, gPad.GetUymin(), gate_min, gPad.GetUymax())
    line_max = TLine(gate_max, gPad.GetUymin(), gate_max, gPad.GetUymax())
    for line in [line_min, line_max]:
        line.Draw()
        line.SetLineWidth(2)
        line.SetLineStyle(2)
        line.SetLineColor(kBlue)
    c1.Update()
    c1.SaveAs('{}/print_integrated_charge_npe.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_waveforms(filename, **kwargs):
    gate_min = kwargs.get('gate_min', None)
    gate_max = kwargs.get('gate_max', None)
    spill = kwargs.get('spill', 1)
    event = kwargs.get('event', 0)
    channels = kwargs.get('channels', [2])
    legend_labels = kwargs.get('legend_labels', ['Cerenkov'])
    legend_ndcs = kwargs.get('legend_ndcs', [0.4, 0.22, 0.53, 0.45])
    legend_text_size = kwargs.get('legend_text_size', None)
    legend_column_count = kwargs.get('legend_column_count', 1)
    colors = kwargs.get('colors', [kBlack])
    y_min = kwargs.get('y_min', None)
    y_max = kwargs.get('y_max', None)
    y_limit = kwargs.get('y_limit', 5000.)
    image_format = kwargs.get('image_format', 'pdf')
    batch_mode = kwargs.get('batch_mode', False)
    x_scale = kwargs.get('x_scale', 0.2)
    x_unit = kwargs.get('x_unit', 'ns')
    grid = kwargs.get('grid', False)
    subdirectory = kwargs.get('subdirectory', '')

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    grs_xs = []
    grs_ys = []
    grs = []
    for channel in channels:
        gr = get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, channel)), x_scale=x_scale)
        grs.append(gr)
        get_graph_min_max(gr)
        gr_x_min, gr_x_max, gr_y_min, gr_y_max = get_graph_min_max(gr)
        grs_xs += [gr_x_min, gr_x_max]
        grs_ys += [gr_y_min, gr_y_max]

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    if grid:
        gPad.SetGrid()

    lg1 = TLegend(legend_ndcs[0], legend_ndcs[1], legend_ndcs[2], legend_ndcs[3])
    set_legend_style(lg1)
    if legend_text_size:
        lg1.SetTextSize(legend_text_size)
    if legend_column_count > 1:
        lg1.SetNColumns(legend_column_count)

    for i, gr in enumerate(grs):
        set_graph_style(gr)
        gr.SetLineColor(colors[i])
        lg1.AddEntry(gr, legend_labels[i], 'l')

        if i == 0:
            gr.Draw('AL')
            # gr.GetXaxis().SetRangeUser(0, 1024)
            # gr.GetXaxis().SetTitle('Time Tick (200 ps)')
            gr.GetXaxis().SetTitle('Time ({})'.format(x_unit))
            gr.GetYaxis().SetTitle('ADC')
            gr.GetYaxis().SetTitleOffset(1.5)
            gr.GetXaxis().SetRangeUser(min(grs_xs), max(grs_xs))
            if y_min is not None and y_max is not None:
                gr.GetYaxis().SetRangeUser(y_min, y_max)
            else:
                y_max = min(max(grs_ys) * 1.1, y_limit)
                gr.GetYaxis().SetRangeUser(min(grs_ys) * 0.9, y_max)
        else:
            gr.Draw('L,sames')

    lg1.Draw()

    if gate_min and gate_max:
        c1.Update()
        line_min = TLine(gate_min, gPad.GetUymin(), gate_min, gPad.GetUymax())
        line_max = TLine(gate_max, gPad.GetUymin(), gate_max, gPad.GetUymax())
        for line in [line_min, line_max]:
            line.Draw()
            line.SetLineWidth(2)
            line.SetLineStyle(2)
            line.SetLineColor(kBlue)
        lg1.AddEntry(line_min, 'gate', 'l')

    figure_dir_subdir = '{}/{}'.format(FIGURE_DIR, subdirectory)
    if not os.path.isdir(figure_dir_subdir):
        os.makedirs(figure_dir_subdir)

    c1.Update()
    c1.SaveAs('{}plot_cerenkov_pulse.{}.spill_{}.event_{}.channel_{}.{}'.format(figure_dir_subdir, filename, spill, event, '_'.join(list(map(str, channels))), image_format))
    if not batch_mode:
        input('Press any key to continue.')


def plot_waveform_hist(filename, **kwargs):
    spill = kwargs.get('spill', 1)
    event = kwargs.get('event', 0)
    channel = kwargs.get('channel', 2)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    gr = get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, channel)))
    h1 = tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, channel))
    h_adc = TH1D('h_adc', 'h_adc', 4096, 0.5, 4096.5)
    for i in range(1, h1.GetNbinsX() + 1):
        h_adc.Fill(h1.GetBinContent(i))

    peak = h_adc.GetBinCenter(h_adc.GetMaximumBin())
    x_min = min(list(gr.GetX()))
    x_max = max(list(gr.GetX()))
    print('peak = {}'.format(peak))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    # gr.Draw('AL')
    # gr.GetXaxis().SetRangeUser(x_min, x_max)
    # tl = TLine(x_min, peak, x_max, peak)
    # tl.SetLineColor(kRed)
    # tl.SetLineWidth(3)
    # tl.Draw()

    set_h1_style(h_adc)
    h_adc.Draw()
    c1.Update()
    tl = TLine(peak, gPad.GetUymin(), peak, gPad.GetUymax())
    tl.SetLineColor(kRed)
    tl.SetLineWidth(3)
    tl.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_waveform_hist.{}.spill_{}.event_{}.channel_{}.pdf'.format(FIGURE_DIR, filename, spill, event, channel))
    input('Press any key to continue.')


def plot_cerenkov_hit_count_per_event(filename):
    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h1 = tf.Get('cerenkovana/fHitCountPerEvent')

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()

    set_h1_style(h1)
    h1.Draw()
    h1.GetXaxis().SetRangeUser(0, 5)
    h1.GetXaxis().SetTitle('Pulse Count per Event')
    h1.GetYaxis().SetTitle('Event Count')
    h1.SetLineColor(kBlack)

    c1.Update()
    draw_statbox(h1, y1=0.7, x1=0.7)

    c1.Update()
    c1.SaveAs('{}/plot_cerenkov_hit_count_per_event.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def plot_cerenkov_adc_spectrum(filename, **kwargs):
    adc_method = kwargs.get('adc_method', 'gate')
    x_min = kwargs.get('x_min', -5.e3)
    x_max = kwargs.get('x_max', 200.e3)
    calibration_constant = kwargs.get('calibration_constant', None)
    log_y = kwargs.get('log_y', False)
    canvas_width = kwargs.get('canvas_width', 800)
    canvas_height = kwargs.get('canvas_height', 800)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h1 = tf.Get('cerenkovana/{}'.format('fHitAdcPerEventByGate' if adc_method == 'gate' else 'fHitAdcPerEvent'))

    if calibration_constant:
        h_calibrate = TH1D('h_calibrate', 'h_calibrate', h1.GetNbinsX(), h1.GetBinLowEdge(1) * calibration_constant, h1.GetBinLowEdge(h1.GetNbinsX() + 1) * calibration_constant)
        for i in range(1, h1.GetNbinsX() + 1):
            h_calibrate.SetBinContent(i, h1.GetBinContent(i))
        h_calibrate.SetEntries(h1.GetEntries())
        h1 = h_calibrate

    h_event_count_per_fragment = tf.Get('cerenkovana/fEventCountPerFragment')
    fragment_count = h_event_count_per_fragment.GetEntries()
    exposure = fragment_count / 60. # hour
    # event_count = h1.Integral(h1.FindBin(5.), h1.GetNbinsX())
    # event_count = h1.Integral(h1.FindBin(50.), h1.GetNbinsX())
    # event_count = h1.Integral(h1.FindBin(20.e3), h1.GetNbinsX())
    event_count = h1.Integral(h1.FindBin(2), h1.GetNbinsX())
    event_rate = event_count / (fragment_count * 40.)
    print('fragment_count = {}'.format(fragment_count))
    print('exposure = {} hour'.format(exposure))
    print('event_count = {}'.format(event_count))
    print('event_rate = {} Hz'.format(event_rate))

    c1 = TCanvas('c1', 'c1', canvas_width, canvas_height)
    set_margin()
    if log_y:
        gPad.SetLogy()
    gPad.SetGrid()

    set_h1_style(h1)
    h1.Draw()
    h1.GetXaxis().SetTitle('ADC per Event')
    if calibration_constant:
        h1.GetXaxis().SetTitle('NPE per Event')
    h1.GetYaxis().SetTitle('Event Count')
    if canvas_height >= 800:
        h1.GetYaxis().SetTitleOffset(2)
    h1.GetXaxis().SetRangeUser(x_min, x_max)
    h1.SetLineColor(kBlack)
    c1.Update()
    draw_statbox(h1, y1=0.7, x1=0.7)

    # h1.GetXaxis().SetRangeUser(4.5e4, 7.e4)
    # spectrum = TSpectrum()
    # spectrum.Search(h1, 3)
    # poly_marker = h1.GetListOfFunctions().FindObject('TPolyMarker')
    # peak_xs = spectrum.GetPositionX()
    # peak_ys = spectrum.GetPositionY()
    # peak_count = spectrum.GetNPeaks()
    # for i in range(peak_count):
    #     print('peak_xs[i] = {}'.format(peak_xs[i]))

    # peak_cut = 4.3e4
    # line = TLine(peak_cut, 0.5, peak_cut, 100)
    # arrow = TArrow(peak_cut, 100, peak_cut + 1e4, 100, 0.03)
    # arrow.SetAngle(60)
    # for ll in [line, arrow]:
    #     ll.SetLineWidth(2)
    #     ll.SetLineStyle(7)
    #     ll.SetLineColor(kRed)
    #     ll.Draw()
    # event_count = h1.Integral(h1.FindBin(peak_cut), h1.GetNbinsX())
    # print('event_count = {}'.format(event_count))

    # tex = TLatex()
    # tex.SetTextFont(43)
    # tex.SetTextSize(25)
    # tex.SetTextAlign(12)
    # tex.DrawLatex(4.25e4, 210, '{:.0f} events'.format(event_count))
    # tex.DrawLatex(5e4, 20, 'peak ADC: {:.1E}'.format(peak_xs[0]))

    c1.Update()
    c1.SaveAs('{}/plot_cerenkov_adc_spectrum.{}.calibrate_{}.log_y_{}.pdf'.format(FIGURE_DIR, filename, True if calibration_constant else False, log_y))
    input('Press any key to continue.')


def plot_cerenkov_adc_spectrum_calibration(filename, **kwargs):
    adc_to_coulomb = 1./4096. / 50 * 0.2e-9 # coulomb

    adc_method = kwargs.get('adc_method', 'gate')
    x_min = kwargs.get('x_min', 5.e3)
    x_max = kwargs.get('x_max', 200.e3)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    h1 = tf.Get('cerenkovana/{}'.format('fHitAdcPerEventByGate' if adc_method == 'gate' else 'fHitAdcPerEvent'))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()

    set_h1_style(h1)
    h1.Draw()
    h1.GetXaxis().SetTitle('ADC per Event')
    h1.GetYaxis().SetTitle('Event Count')
    h1.GetXaxis().SetRangeUser(x_min, x_max)
    h1.SetLineColor(kBlack)
    c1.Update()
    draw_statbox(h1, y1=0.78, x1=0.6)

    h1.Fit('gaus')
    npe = (h1.GetMean() / h1.GetRMS())**2
    calibration_constant = npe / h1.GetMean()
    gain = h1.GetMean() * adc_to_coulomb / (npe * 1.602e-19)

    print('npe = {}'.format(npe))
    print('calibration_constant = {}'.format(calibration_constant))
    print('gain = {}'.format(gain))

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(43)
    latex.SetTextSize(28)
    latex.DrawLatex(0.5, 0.67, 'NPE = {:.1f}'.format(npe))
    latex.DrawLatex(0.5, 0.60, 'calib. = {:.1E} PE / ADC'.format(calibration_constant))
    latex.DrawLatex(0.5, 0.53, 'gain = {:.1E}'.format(gain))

    c1.Update()
    c1.SaveAs('{}/plot_cerenkov_adc_spectrum_calibration.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_cerenkov_calibration_constant_vs_light_level():
    filenames = [
        'cerenkovana.calibration_run_5.root',
        'cerenkovana.calibration_run_1.root',
        'cerenkovana.calibration_run_2.root',
        'cerenkovana.calibration_run_3.root',
        'cerenkovana.calibration_run_4.root',
    ]

    pulse_widths = [9.2, 9.4, 9.6, 9.8, 10.0]

    calibration_constants = []
    adc_means = []
    npes = []
    for filename in filenames:
        tf = TFile('{}/{}'.format(DATA_DIR, filename))
        h1 = tf.Get('cerenkovana/fHitAdcPerEventByGate')
        h1.GetXaxis().SetRangeUser(5e3, 200.e3)

        mean = h1.GetMean()
        sigma = h1.GetRMS()
        # f1 = TF1('f1', 'gaus')
        # h1.Fit('f1')
        # mean = f1.GetParameter(1)
        # sigma = f1.GetParameter(2)

        npe = (mean / sigma)**2
        calibration_constant = npe / mean
        npes.append(npe)
        calibration_constants.append(calibration_constant)
        adc_means.append(h1.GetMean())
        tf.Close()

    gr = TGraph(len(adc_means), np.array(adc_means), np.array(calibration_constants))
    # gr = TGraph(len(npes), np.array(npes), np.array(calibration_constants))
    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    set_graph_style(gr)
    gr.Draw('AP')
    gr.GetYaxis().SetMaxDigits(3)
    gr.GetYaxis().SetRangeUser(3e-3, 3.7e-3)
    gr.GetXaxis().SetTitle('Mean ADC')
    gr.GetYaxis().SetTitle('NPE / ADC')
    gr.Fit('pol0')

    gStyle.SetStatW(0.26)
    gStyle.SetStatH(0.25)

    c1.Update()
    c1.SaveAs('{}/plot_cerenkov_calibration_constant_vs_light_level.pdf'.format(FIGURE_DIR))
    input('Press any key to continue.')


def compare_cerenkov_adc_spectra(filenames, **kwargs):
    adc_method = kwargs.get('adc_method', 'gate')
    x_min = kwargs.get('x_min', -5.e3)
    x_max = kwargs.get('x_max', 200.e3)
    calibration_constant = kwargs.get('calibration_constant', None)
    log_y = kwargs.get('log_y', False)
    colors = kwargs.get('colors', [kRed, kBlue, kGreen + 2])
    legend_texts = kwargs.get('legend_texts', ['a', 'b', 'c'])

    h1s = []
    for i, filename in enumerate(filenames):
        tf = TFile('{}/{}'.format(DATA_DIR, filename))
        h1 = tf.Get('cerenkovana/{}'.format('fHitAdcPerEventByGate' if adc_method == 'gate' else 'fHitAdcPerEvent'))
        h1.SetDirectory(0)

        h_event_count_per_fragment = tf.Get('cerenkovana/fEventCountPerFragment')
        fragment_count = h_event_count_per_fragment.GetEntries()
        exposure = fragment_count / 60. # hour
        event_count = h1.Integral()
        event_rate = event_count / (fragment_count * 40.)

        print('\nlegend_texts[i] = {}'.format(legend_texts[i]))
        print('exposure = {} hour'.format(exposure))
        print('event_count = {}'.format(event_count))
        print('fragment_count = {}'.format(fragment_count))
        print('event_rate = {} Hz'.format(event_rate))

        # h1.Scale(1. / h1.Integral())
        # h1.Scale(1. / h1.GetMaximum())
        h1.Scale(1. / exposure)
        h1s.append(h1)

    print('len(h1s) = {}'.format(len(h1s)))

    if calibration_constant:
        for j, h1 in enumerate(h1s):
            h_calibrate = TH1D('h_calibrate', 'h_calibrate', h1.GetNbinsX(), h1.GetBinLowEdge(1) * calibration_constant, h1.GetBinLowEdge(h1.GetNbinsX() + 1) * calibration_constant)
            for i in range(1, h1.GetNbinsX() + 1):
                h_calibrate.SetBinContent(i, h1.GetBinContent(i))
                h_calibrate.SetEntries(h1.GetEntries())
            h1s[j] = h_calibrate

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    if log_y:
        gPad.SetLogy()
    gPad.SetGrid()

    y_max = 0.
    for i, h1 in enumerate(h1s):
        if h1.GetMaximum() > y_max:
            y_max = h1.GetMaximum()

    lg1 = TLegend(0.35, 0.88 - 0.06 * len(filenames), 0.55, 0.88)
    set_legend_style(lg1)

    for i, h1 in enumerate(h1s):
        set_h1_style(h1)
        h1.SetLineColor(colors[i])
        lg1.AddEntry(h1, legend_texts[i], 'l')

        if i == 0:
            h1.Draw()
            h1.GetXaxis().SetTitle('ADC per Event')
            if calibration_constant:
                h1.GetXaxis().SetTitle('NPE per Event')
            h1.GetYaxis().SetTitle('Event Count / Running Hour')
            h1.GetYaxis().SetTitleOffset(1.5)
            h1.GetXaxis().SetRangeUser(x_min, x_max)
            h1.GetXaxis().SetRangeUser(x_min, x_max)
            h1.GetYaxis().SetRangeUser(1.e-1, y_max * 1.1)
            # c1.Update()
            # draw_statbox(h1, y1=0.7, x1=0.7)
        h1.Draw('sames')

    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/compare_cerenkov_adc_spectra.{}.calibrate_{}.log_y_{}.pdf'.format(FIGURE_DIR, filename, True if calibration_constant else False, log_y))
    c1.SaveAs('{}/compare_cerenkov_adc_spectra.{}.calibrate_{}.log_y_{}.png'.format(FIGURE_DIR, filename, True if calibration_constant else False, log_y))
    input('Press any key to continue.')


def get_pulse_spill_numbers(filename, **kwargs):
    calibration_constant = kwargs.get('calibration_constant', 1.)

    spill_numbers = []
    subrun_number_pulse_adcs = []

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    for pulse in tf.Get('cerenkovana/fPulseTree'):
        spill_numbers.append(pulse.fSubRunNumber)
        subrun_number_pulse_adcs.append([pulse.fSubRunNumber, pulse.fPulseAdc])

    subrun_number_pulse_adcs = sorted(subrun_number_pulse_adcs, key=lambda x: x[1])
    for subrun_number_pulse_adc in subrun_number_pulse_adcs:
        subrun_number = subrun_number_pulse_adc[0]
        pulse_adc = subrun_number_pulse_adc[1]
        pulse_npe = subrun_number_pulse_adc[1] * calibration_constant
        print('subrun_number = {}, adc = {}, pulse_npe = {}'.format(subrun_number, pulse_adc, pulse_npe))

    return spill_numbers


def plot_dndx_vs_time_of_flight():
    # index_of_refraction = 1.0004 # CO2
    # index_of_refraction = 1.000035 # helium
    index_of_refraction = 1.0003 # N2

    cerenkov_length = 2.925    # m
    trigger_distance = 4.6      # m
    pmt_quantum_efficiency = 0.2
    time_of_flight_max = trigger_distance / (SPEED_OF_LIGHT / index_of_refraction) * 1.e9 # ns
    time_of_flight_min = trigger_distance / SPEED_OF_LIGHT * 1.e9 # ns
    time_of_flight_count = 100.
    time_of_flight_delta = (time_of_flight_max - time_of_flight_min) / time_of_flight_count
    time_of_flight_epsilon = time_of_flight_min / 1.e5

    print('time_of_flight_min = {}'.format(time_of_flight_min))
    print('time_of_flight_max = {}'.format(time_of_flight_max))
    print('time_of_flight_delta = {}'.format(time_of_flight_delta))

    time_of_flights = []        # ns
    dns = []
    dndxs = []
    momentums = []
    for time_of_flight in np.arange(time_of_flight_min + time_of_flight_epsilon, time_of_flight_max, time_of_flight_delta):
        time_of_flights.append(time_of_flight) # ns

        time_of_flight *= 1.e-9 # s
        beta = (trigger_distance / time_of_flight) / SPEED_OF_LIGHT
        gamma = 1. / (1 - beta**2)**0.5
        mass = PDG.GetParticle('mu+').Mass()
        momentum = beta * gamma * mass # GeV
        momentums.append(momentum)

        theta = math.acos(1. / index_of_refraction / beta) * 180. / pi
        sin_square_theta = 1. - (1. / index_of_refraction / beta)**2.
        dndx = 2. * pi * 1. / 137. * sin_square_theta * pmt_quantum_efficiency * (1. / 300 - 1. / 500.) * 1.e9
        dn = dndx * cerenkov_length
        dndxs.append(dndx)
        dns.append(dn)

    # gr1 = TGraph(len(dns), np.array(time_of_flights), np.array(dndxs))
    gr_tof = TGraph(len(dns), np.array(time_of_flights), np.array(dns))
    gr_gev = TGraph(len(dns), np.array(momentums), np.array(dns))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_graph_style(gr_tof)
    gr_tof.Draw('AL')
    gr_tof.GetXaxis().SetTitle('Time of Flight (ns)')
    gr_tof.GetYaxis().SetTitle('NPE')
    c1.Update()
    c1.SaveAs('{}/plot_dndx_vs_time_of_flight.tof.pdf'.format(FIGURE_DIR))

    c2 = TCanvas('c2', 'c2', 800, 600)
    set_margin()
    gPad.SetGrid()
    set_graph_style(gr_gev)
    gr_gev.Draw('AL')
    gr_gev.GetXaxis().SetTitle('Muon Momentum (GeV)')
    gr_gev.GetYaxis().SetTitle('NPE')
    c2.Update()
    c2.SaveAs('{}/plot_dndx_vs_time_of_flight.gev.pdf'.format(FIGURE_DIR))

    input('Press any key to continue.')

    # print('theta = {} degree'.format(theta))
    # print('sin_square_theta = {}'.format(sin_square_theta))
    # print('dndx = {}'.format(dndx))
    # print('dn = {}'.format(dn))


def get_test_beam_chamber_volume():
    length = 75. * 12. * INCH_TO_METER
    width = (23. * 12. + 6.) * INCH_TO_METER
    height = 95. * INCH_TO_METER

    area = pi * (width / 2.)**2 / 2. + width * height
    volume = area * length

    print('INCH_TO_METER = {}'.format(INCH_TO_METER))
    print('length = {}'.format(length))
    print('width = {}'.format(width))
    print('height = {}'.format(height))
    print('area = {} m2'.format(area))
    print('volume = {} m3'.format(volume))
    print('volume = {} cubic feet'.format(volume * 35.3147))

    cerenkov_radius = 0.15
    cerenkov_length = 3.
    cerenkov_volume = pi * cerenkov_radius**2 * cerenkov_length
    # cerenkov_volume = 0.25
    print('cerenkov_volume = {}'.format(cerenkov_volume))
    print('cerenkov_volume / volume = {}'.format(cerenkov_volume / volume))


def sort_cerenkov_event_by_adc(filename, **kwargs):
    calibration_constant = kwargs.get('calibration_constant', None)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))

    subrun_number_adcs = []
    for event in tf.Get('cerenkovana/fPulseTree'):
        # print('event.fSubRunNumber = {}'.format(event.fSubRunNumber))
        # print('event.fPulseAdc = {}'.format(event.fPulseAdc))
        subrun_number_adcs.append([event.fSubRunNumber, event.fPulseAdc])

    subrun_number_adcs = sorted(subrun_number_adcs, key=lambda x: x[0], reverse=True)
    # pprint(subrun_number_adcs)
    for subrun_number_adc in subrun_number_adcs:
        subrun_number = subrun_number_adc[0]
        adc = subrun_number_adc[1]
        npe = None
        if calibration_constant:
            npe = adc * calibration_constant
        print('subrun_number = {}, adc = {}, npe = {}'.format(subrun_number, adc, npe))


def plot_trigger_count_per_fragment(filename, **kwargs):
    x_axis_title = kwargs.get('x_axis_title', 'Time (minute)')
    y_axis_title = kwargs.get('y_axis_title', 'Coincidence Count per Minute')
    y_min = kwargs.get('y_min', None)
    y_max = kwargs.get('y_max', None)
    grid = kwargs.get('grid', False)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))

    spill_number_trigger_counts = []

    keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
    for key in keys:
        print('key = {}'.format(key))
        spill_number = int(key[5:])

        gDirectory.cd(key)
        sub_keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
        trigger_count = len(sub_keys) - 1
        # if len(sub_keys) > 1:
        #     print('sub_keys = {}'.format(sub_keys))
        spill_number_trigger_counts.append([spill_number, trigger_count])
        gDirectory.cd('..')
        # break

    spill_number_trigger_counts = sorted(spill_number_trigger_counts, key=lambda x: x[0])
    spill_numbers = [float(spill_number_trigger_count[0]) for spill_number_trigger_count in spill_number_trigger_counts]
    trigger_counts = [float(spill_number_trigger_count[1]) for spill_number_trigger_count in spill_number_trigger_counts]

    h1 = TH1D('h1', 'h1', int(max(spill_numbers) - min(spill_numbers)) + 1, min(spill_numbers) - 0.5, max(spill_numbers) + 0.5)
    for i in range(len(trigger_counts)):
        h1.Fill(spill_numbers[i], trigger_counts[i])
    if y_min is not None and y_max is not None:
        h1.GetYaxis().SetRangeUser(y_min, y_max)

    print('h1.Integral() = {}'.format(h1.Integral()))
    print('h1.Integral() = {}'.format(h1.Integral(2, h1.GetNbinsX())))
    print('h1.GetNbinsX() = {}'.format(h1.GetNbinsX()))
    print('h1.Integral() / h1.GetNbinsX() = {}'.format(float(h1.Integral()) / h1.GetNbinsX()))

    gr = TGraph(len(spill_numbers), np.array(spill_numbers), np.array(trigger_counts))

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    gPad.SetLogy()
    if grid:
        gPad.SetGrid()

    # set_graph_style(gr)
    # gr.Draw('APL')

    set_h1_style(h1)
    h1.Draw('hist')
    h1.GetXaxis().SetTitle(x_axis_title)
    h1.GetYaxis().SetTitle(y_axis_title)

    c1.Update()
    c1.SaveAs('{}/plot_trigger_count_per_fragment.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_coincidence_us_tof_2wcs(filename, **kwargs):
    subrun_count = kwargs.get('subrun_count', 4000)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))

    # h1 = TH1D('h1', 'h1', 3531, 0.5, 3531.5)
    h1 = TH1D('h1', 'h1', subrun_count, 0.5, subrun_count + 0.5)

    for event in tf.Get('cerenkovana/fTofTree'):
        h1.Fill(event.fSubRunNumber)

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()
    set_h1_style(h1)
    h1.GetXaxis().SetTitle('Time (minute)')
    h1.GetYaxis().SetTitle('Event Count per Minute')
    h1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_coincidence_us_tof_2wcs.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')


def plot_coincidence_us_tof_2wcs_multiple_runs():
    filenames = ['cerenkovana.cosmic_run_2499.root', 'cerenkovana.cosmic_run_2500.root', 'cerenkovana.cosmic_run_2501.root']
    run_numbers = [2499, 2500, 2501]
    # Fri Mar 22 13:22:09 CDT 2019: START transition complete for run 2499
    # Fri Mar 22 16:39:01 CDT 2019: STOP transition complete for run 2499

    # Fri Mar 22 16:41:02 CDT 2019: START transition complete for run 2500
    # Fri Mar 22 16:55:56 CDT 2019: STOP transition complete for run 2500

    # Fri Mar 22 17:19:27 CDT 2019: START transition complete for run 2501
    # Mon Mar 25 08:27:15 CDT 2019: STOP transition complete for run 2501

    start_times = [
        datetime(2019, 3, 22, 13, 22, 9),
        datetime(2019, 3, 22, 16, 41, 2),
        datetime(2019, 3, 22, 17, 19, 27)
    ]

    end_times = [
        datetime(2019, 3, 22, 16, 39, 1),
        datetime(2019, 3, 22, 16, 55, 56),
        datetime(2019, 3, 25, 8, 27, 15)
    ]

    subrun_counts = [196, 14, 3531]

    gaps = [
        0,
        int((start_times[1] - end_times[0]).total_seconds() / 60.),
        int((start_times[2] - end_times[1]).total_seconds() / 60.),
    ]

    minute_count = sum(subrun_counts) + sum(gaps)
    h1 = TH1D('h1', 'h1', minute_count, 0.5, minute_count + 0.5)
    print('gaps = {}'.format(gaps))

    tf0 = TFile('{}/cerenkovana.cosmic_run_{}.root'.format(DATA_DIR, run_numbers[0]))
    for event in tf0.Get('cerenkovana/fTofTree'):
        minute = gaps[0] + event.fSubRunNumber
        h1.Fill(minute)

    tf1 = TFile('{}/cerenkovana.cosmic_run_{}.root'.format(DATA_DIR, run_numbers[1]))
    for event in tf1.Get('cerenkovana/fTofTree'):
        minute = gaps[0] + subrun_counts[0] + gaps[1] + event.fSubRunNumber
        h1.Fill(minute)

    tf2 = TFile('{}/cerenkovana.cosmic_run_{}.root'.format(DATA_DIR, run_numbers[2]))
    for event in tf2.Get('cerenkovana/fTofTree'):
        minute = gaps[0] + subrun_counts[0] + gaps[1] + subrun_counts[1] + gaps[2] + event.fSubRunNumber
        h1.Fill(minute)

    cosmic_ray_rate = h1.Integral(500, 3000) / ((3000 - 500 + 1) * 60)
    print('cosmic_ray_rate = {}'.format(cosmic_ray_rate))

    # c1 = TCanvas('c1', 'c1', 800, 600)
    # set_margin()
    # set_h1_style(h1)
    # h1.GetXaxis().SetTitle('Time (minute)')
    # h1.GetYaxis().SetTitle('Event Count per Minute')
    # h1.Draw()
    # c1.Update()
    # c1.SaveAs('{}/plot_coincidence_us_tof_2wcs_multiple_runs.pdf'.format(FIGURE_DIR))

    # c2 = TCanvas('c2', 'c2', 800, 600)
    # set_margin()
    # set_h1_style(h1)
    # h1.GetXaxis().SetTitle('Time (minute)')
    # h1.GetYaxis().SetTitle('Event Count per Minute')
    # h1.GetXaxis().SetRangeUser(150, 300)
    # h1.Draw()
    # c2.Update()
    # tls = [
    #     TLine(gaps[0] + subrun_counts[0], gPad.GetUymin(), gaps[0] + subrun_counts[0], gPad.GetUymax()),
    #     TLine(gaps[0] + subrun_counts[0] + gaps[1], gPad.GetUymin(), gaps[0] + subrun_counts[0] + gaps[1], gPad.GetUymax()),
    #     TLine(gaps[0] + subrun_counts[0] + gaps[1] + subrun_counts[1], gPad.GetUymin(), gaps[0] + subrun_counts[0] + gaps[1] + subrun_counts[1], gPad.GetUymax()),
    #     TLine(gaps[0] + subrun_counts[0] + gaps[1] + subrun_counts[1] + gaps[2], gPad.GetUymin(), gaps[0] + subrun_counts[0] + gaps[1] + subrun_counts[1] + gaps[2], gPad.GetUymax()),
    # ]
    # tl_colors = [kRed, kGreen + 1, kGreen + 1, kMagenta + 1]
    # for i, tl in enumerate(tls):
    #     tl.SetLineStyle(3)
    #     tl.SetLineWidth(3)
    #     tl.SetLineColor(tl_colors[i])
    #     tl.Draw()
    # latex = TLatex()
    # latex.SetNDC()
    # latex.SetTextFont(43)
    # latex.SetTextSize(22)
    # latex.SetTextColor(kRed)
    # latex.DrawLatex(0.25, 0.83, '#splitline{run}{2499}')
    # latex.SetTextColor(kGreen + 1)
    # latex.DrawLatex(0.4, 0.83, '#splitline{run}{2500}')
    # latex.SetTextColor(kMagenta + 1)
    # latex.DrawLatex(0.72, 0.83, '#splitline{run}{2501}')
    # c2.Update()
    # c2.SaveAs('{}/plot_coincidence_us_tof_2wcs_multiple_runs.beam.pdf'.format(FIGURE_DIR))

    c3 = TCanvas('c3', 'c3', 800, 600)
    set_margin()
    set_h1_style(h1)
    h1.GetXaxis().SetTitle('Time (hour)')
    h1.GetYaxis().SetTitle('Event Count per Hour')
    h1.Rebin(60)
    h1.GetXaxis().SetRangeUser(500, 3000)
    h1.Draw()
    c3.Update()
    c3.SaveAs('{}/plot_coincidence_us_tof_2wcs_multiple_runs.cosmic.pdf'.format(FIGURE_DIR))

    input('Press any key to continue.')


def plot_us_tof_waveform(filename, **kwargs):
    spill = kwargs.get('spill', 1)
    event = kwargs.get('event', 0)
    channel = kwargs.get('channel', 2)

    tf = TFile('{}/{}'.format(DATA_DIR, filename))
    grs = [
        get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, 3))),
        get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, 4))),
        get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, 5))),
        get_graph_from_th1(tf.Get('Spill{}/Event{}/Channel{}'.format(spill, event, 6)))
    ]
    colors = [kBlack, kBlue, kRed, kGreen + 1]
    legend_names = ['channel 1', 'channel 2', 'channel 3', 'channel 4']

    c1 = TCanvas('c1', 'c1', 800, 600)
    set_margin()

    lg1 = TLegend(0.6, 0.24, 0.84, 0.55)
    set_legend_style(lg1)

    for i, gr in enumerate(grs):
        set_graph_style(gr)
        gr.SetLineColor(colors[i])
        if i == 0:
            gr.GetXaxis().SetTitle('Time (0.2 ns)')
            gr.GetYaxis().SetTitle('ADC')
            gr.GetXaxis().SetRangeUser(0, 1024)
            gr.GetYaxis().SetTitleOffset(1.5)
            gr.Draw('AL')
        else:
            gr.Draw('L,sames')
        lg1.AddEntry(gr, legend_names[i], 'l')

    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_us_tof_waveform.{}.spill_{}.event_{}.pdf'.format(FIGURE_DIR, filename, spill, event))
    input('Press any key to continue.')


def get_subrun_number_event_indexs(run_number, **kwargs):
    tree_name = kwargs.get('tree_name', 'fPulseTree')
    config = kwargs.get('config', '')

    tf = TFile('{}/cerenkovana.beam_run_{}{}.root'.format(DATA_DIR, run_number, config))
    subrun_number_event_indexs = []
    for event in tf.Get('cerenkovana/{}'.format(tree_name)):
        subrun_number_event_indexs.append((event.fSubRunNumber, event.fEventIndex))

    return subrun_number_event_indexs


def plot_alignment_data():
    with open('data/alignment/NTB summary_up_ct_dn.txt') as f_txt:
        for row in csv.reader(f_txt, delimiter=','):
            # print('row = {}'.format(row))
            detector = row[1].strip()
            x = float(row[2])
            y = float(row[3])
            z = float(row[4])

            if '_CT' in detector:
                print('detector = {}'.format(detector))
                print('x = {}, y = {}, z = {}'.format(x, y, z))


def read_wire_chamber_position_rotation():
    detector_positions = {}

    with open('data/alignment/NTB summary_up_ct_dn.txt') as f_txt:
        for row in csv.reader(f_txt, delimiter=','):
            detector_name = row[1].strip()
            x = float(row[2])
            y = float(row[3])
            z = float(row[4])

            if '_CT' in detector_name:
                if 'NTB-TGT-COLL-002-TOF-1' in detector_name:
                    detector = 'tof_us'
                elif 'NTB-CERENKOV-TOF-1' in detector_name:
                    detector = 'tof_ds_pmt'
                elif 'NTB-CERENKOV-TOF-2' in detector_name:
                    detector = 'tof_ds_sipm'
                elif 'NTB-MWPC-AK' in detector_name:
                    detector = 'wc_1'
                elif 'NTB-MWPC-AL' in detector_name:
                    detector = 'wc_2'
                elif 'NTB-MWPC-AF' in detector_name:
                    detector = 'wc_3'
                elif 'NTB-MWPC-AI' in detector_name:
                    detector = 'wc_4'

                detector_positions[detector] = {
                    'x': -x,
                    'y': z,
                    'z': y
                }
    return detector_positions


def plot_good_particle_position_mwpc():
    tf = TFile('merge_tree.root')
    tf.cd()
    key_names = [key.GetName() for key in gDirectory.GetListOfKeys()]

    tree = tf.Get(key_names[0])
    tree.Print()
    for event in tf.Get(key_names[0]):
        print('event.SpillID = {}'.format(event.SpillID))
        print('event.EventID = {}'.format(event.EventID))
        # event.xwire_chamber_4_detector


# 20190531_testbeam_good_particle_position
plot_good_particle_position_mwpc()

# 20190502_testbeam_scintillator_paddle_beam
# gStyle.SetOptStat(0)
# plot_trigger_count_per_fragment('V1742Analysis.run_2724.root', x_axis_title='Spill Number', y_axis_title='Event Count', grid=True)
# plot_trigger_count_per_fragment('V1742Analysis.run_2717.root', x_axis_title='Spill Number', y_axis_title='Event Count', grid=True)
# plot_trigger_count_per_fragment('V1742Analysis.run_2627.root', x_axis_title='Spill Number', y_axis_title='Event Count', grid=True)
# run_number = 2717
# config = ''
# # subrun_number_event_indexs = get_subrun_number_event_indexs(run_number, config=config)
# subrun_number_event_indexs = [
#     (98, 23)
# ]
# run_number = 2724
# config = '.v1742EventIndex_i'
# config = '.v1742EventIndex_i+1'
# config = '.no_wire_chamber'
# subrun_number_event_indexs = get_subrun_number_event_indexs(run_number, config=config)
# config = '.v1742EventIndex_i+1.timing.80ns'
# subrun_number_event_indexs = [
#     (338, 6)
# ]
# subrun_number_event_indexs = [
#     # (137, 5),
#     (167, 1),
#     (323, 6),
#     # (33, 6),
#     (352, 6),
#     # (366, 4),
#     (414, 4),
#     (432, 4),
#     (443, 6),
#     # (91, 6)
# ]
# config = '.v1742EventIndex_i.timing'
# subrun_number_event_indexs = [
#     # (335, 2),
#     (392, 2)
# ]
# config = '.missing_paddle'
# subrun_number_event_indexs = [
#     (8, 3)
# ]
# for (subrun_number, event_index) in subrun_number_event_indexs:
#     print('subrun_number = {}'.format(subrun_number))
#     print('event_index = {}'.format(event_index))
#     plot_waveforms(
#         'V1742Analysis.run_{}.root'.format(run_number),
#         spill=subrun_number, event=event_index,
#         channels=[3,
#                   4, 5, 6, 7,
#                   8, 9, 10, 11,
#                   12, 13, 14, 15],
#         legend_labels=['Cerenkov',
#                        'Paddle 1', 'Paddle 2', 'Paddle 3', 'Paddle 4',
#                        'US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4',
#                        'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'DS TOF 4'],
#         legend_ndcs=[0.65, 0.18, 0.85, 0.55],
#         legend_text_size=20,
#         colors=COLORS,
#         batch_mode=True,
#         image_format='pdf',
#         subdirectory='run_{}{}/'.format(run_number, config)
#     )
#     # break

# 20190424_testbeam_alignment
# plot_alignment_data()

# 20190419_testbeam_scintillator_paddle_by_mwpc
# print('len(COLORS = {}'.format(len(COLORS)))
# subrun_number_event_indexs = get_subrun_number_event_indexs(2690)
# for (subrun_number, event_index) in subrun_number_event_indexs:
#     print('subrun_number = {}'.format(subrun_number))
#     print('event_index = {}'.format(event_index))
#     plot_waveforms(
#         'V1742Analysis.run_2690.root',
#         spill=subrun_number, event=event_index,
#         channels=[3,
#                   4, 5, 6, 7,
#                   8, 9, 10, 11,
#                   12, 13, 14, 15],
#         legend_labels=['Cerenkov',
#                        'Paddle 1', 'Paddle 2', 'Paddle 3', 'Paddle 4',
#                        'US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4',
#                        'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'DS TOF 4'],
#         # channels=[3, 4, 5, 6, 7, 12, 13, 14, 15],
#         # legend_labels=['Cerenkov', 'Paddle 1', 'Paddle 2', 'Paddle 3', 'Paddle 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'DS TOF 4'],
#         # channels=[3, 4, 5, 6, 7, 8, 9, 10, 11],
#         # legend_labels=['Cerenkov', 'Paddle 1', 'Paddle 2', 'Paddle 3', 'Paddle 4', 'US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4'],
#         legend_ndcs=[0.65, 0.18, 0.85, 0.55],
#         legend_text_size=20,
#         # legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#         # legend_column_count=4,
#         # channels=[6, 7],
#         # legend_labels=['Paddle 3', 'Paddle 4'],
#         # legend_ndcs=[0.6, 0.2, 0.8, 0.4],
#         # legend_text_size=28,
#         # grid=True,
#         colors=COLORS,
#         # x_scale=1.,
#         # x_unit='TDC',
#         batch_mode=True,
#     )
#     # break
# plot_waveforms('V1742Analysis.run_2689.root',
#                spill=73, event=1,
#                # channels=[3, 4, 5, 6, 7, 12, 13, 14, 15],
#                channels=[3, 4, 5, 6, 7, 8, 9, 10, 11],
#                legend_labels=['Cerenkov', 'Paddle 1', 'Paddle 2', 'Paddle 3', 'Paddle 4', 'US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS,
#                x_scale=1.,
#                x_unit='TDC')

# 20190401_testbeam_beam_cerenkov
# plot_waveforms('V1742Analysis.run_2645.root',
#                spill=10, event=4,
#                channels=[8, 9, 10, 11, 12, 13, 14, 15],
#                legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'Cerenkov'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS,
#                x_scale=1.,
#                x_unit='TDC')
# plot_cerenkov_pulse('V1742Analysis.run_2626.root', spill=55, event=2, gate_min=150, gate_max=800, channel=15, legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_cerenkov_pulse('V1742Analysis.run_2626.root', spill=55, event=2, gate_min=150, gate_max=800, channel=15, legend_ndcs=[0.5, 0.22, 0.6, 0.35], trigger_channels=[11, 12])
# plot_cerenkov_pulse('V1742Analysis.run_2626.root', spill=55, event=2, gate_min=150, gate_max=800, channel=15, legend_ndcs=[0.5, 0.22, 0.6, 0.35], trigger_channels=[11, 12])
# plot_cerenkov_pulse('V1742Analysis.run_2627.root', spill=1, event=1, gate_min=150, gate_max=800, channel=8, legend_ndcs=[0.5, 0.22, 0.6, 0.35], trigger_channels=[9, 10])
# plot_cerenkov_pulse('V1742Analysis.run_2627.root', spill=1, event=7, gate_min=150, gate_max=800, channel=8, legend_ndcs=[0.5, 0.22, 0.6, 0.35], trigger_channels=[12, 15])
# plot_waveforms('V1742Analysis.run_2626.root', spill=55, event=2, gate_min=150, gate_max=800, channels=[15, 11, 14], colors=[kBlack, kRed, kBlue], legend_labels=['Cerenkov', 'US TOF', 'DS TOF'], legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_waveforms('V1742Analysis.run_2626.root', spill=53, event=19, channels=[15, 11, 14], colors=[kBlack, kRed, kBlue], legend_labels=['Cerenkov', 'US TOF', 'DS TOF'], legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_waveforms('V1742Analysis.run_2627.root',
#                spill=1, event=10,
#                # spill=3, event=14,
#                # spill=4, event=12,
#                # spill=4, event=19,
#                # spill=9, event=13,
#                # spill=5, event=19,
#                channels=[8, 9, 10, 11, 12, 13, 14, 15],
#                legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'Cerenkov'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS,
#                image_format='png')
# plot_waveforms('V1742Analysis.run_2626.root',
#                # spill=13, event=4,
#                # spill=16, event=10,
#                # spill=20, event=1,
#                # spill=22, event=10,
#                # spill=29, event=3,
#                # spill=29, event=8,
#                # spill=30, event=4,
#                # spill=35, event=4,
#                spill=55, event=2,
#                channels=[8, 9, 10, 11, 12, 13, 14, 15],
#                legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'Cerenkov'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS)
# plot_waveforms('V1742Analysis.run_2634.root',
#                spill=1, event=26,
#                # spill=1, event=0,
#                # spill=1, event=1,
#                # spill=3, event=16,
#                channels=[8, 9, 10, 11, 12, 13, 14, 15, 20],
#                legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'DS TOF 4', 'Cerenkov'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS)
# spill_events = [
#     # (3, 11),
#     (3, 14),
#     (3, 17),
#     (9, 13),
#     (10, 8),
#     # (10, 3)
#     # (4, 12)
# ]
# for spill_event in spill_events:
#     spill = spill_event[0]
#     event = spill_event[1]
#     plot_waveforms('V1742Analysis.run_2627.root',
#                    spill=spill, event=event,
#                    channels=[8, 9, 10, 11, 12, 13, 14, 15],
#                    legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'Cerenkov'],
#                    legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                    colors=COLORS,
#                    image_format='pdf',
#                    batch_mode=True)
# plot_waveforms('V1742Analysis.run_2627.root',
#                # spill=3, event=14,
#                # spill=3, event=17,
#                # spill=9, event=13,
#                spill=10, event=8,
#                # spill=11, event=12,
#                # spill=3, event=12,
#                # spill=3, event=12,
#                # spill=11, event=7,
#                # spill=3, event=1,
#                channels=[8, 9, 10, 11, 12, 13, 14, 15],
#                legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'Cerenkov'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS,
#                # image_format='png',
#                batch_mode=False)
# print_integrated_charge_npe('V1742Analysis.run_2627.root', spill=10, event=8, channel=15, gate_min=350, gate_max=550, calibration_constant=3.362e-3)
# print_integrated_charge_npe('V1742Analysis.run_2627.root', spill=3, event=14, channel=15, gate_min=400, gate_max=600, calibration_constant=3.362e-3)
# print_integrated_charge_npe('V1742Analysis.run_2627.root', spill=9, event=13, channel=15, gate_min=300, gate_max=500, calibration_constant=3.362e-3)
# plot_waveforms('V1742Analysis.run_2628.root',
#                spill=6, event=7,
#                # spill=9, event=25,
#                # spill=18, event=23,
#                # spill=24, event=2,
#                channels=[8, 9, 10, 11, 12, 13, 14, 15],
#                legend_labels=['US TOF 1', 'US TOF 2', 'US TOF 3', 'US TOF 4', 'DS TOF 1', 'DS TOF 2', 'DS TOF 3', 'Cerenkov'],
#                legend_ndcs=[0.6, 0.18, 0.8, 0.55],
#                colors=COLORS)
# plot_waveform_hist('V1742Analysis.run_2627.root', spill=1, event=10, channel=9)

# 20190325_testbeam_beam_tof
# gStyle.SetOptStat(0)
# plot_coincidence_us_tof_2wcs('cerenkovana.cosmic_run_2499.root', subrun_count=196)
# plot_coincidence_us_tof_2wcs('cerenkovana.cosmic_run_2500.root', subrun_count=14)
# plot_coincidence_us_tof_2wcs('cerenkovana.cosmic_run_2501.root', subrun_count=3531)
# plot_coincidence_us_tof_2wcs_multiple_runs()
# plot_us_tof_waveform('V1742Analysis.run_2500.root', spill=1, event=0)
# plot_us_tof_waveform('V1742Analysis.run_2500.root', spill=1, event=5)
# plot_us_tof_waveform('V1742Analysis.run_2500.root', spill=1, event=7)
# plot_us_tof_waveform('V1742Analysis.run_2500.root', spill=1, event=10)
# plot_us_tof_waveform('V1742Analysis.run_2500.root', spill=1, event=11)

# 20190322_testbeam_cerenkov_mwpc_trigger
# gStyle.SetOptStat('emr')
# gStyle.SetOptStat(0)
# get_test_beam_chamber_volume()
# plot_cerenkov_pulse('V1742Analysis.run_2484.root', spill=243, event=1, gate_min=150, gate_max=800, channel=7, legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_cerenkov_pulse('V1742Analysis.run_2484.root', spill=502, event=1, gate_min=150, gate_max=800, channel=7, legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_cerenkov_pulse('V1742Analysis.run_2484.root', spill=765, event=0, gate_min=150, gate_max=800, channel=7, y_min=2100, y_max=2200, legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_cerenkov_pulse('V1742Analysis.run_2484.root', spill=51, event=0, gate_min=150, gate_max=800, channel=7, y_min=2000, y_max=2200, legend_ndcs=[0.5, 0.22, 0.6, 0.35])
# plot_trigger_count_per_fragment('V1742Analysis.run_2484.root')
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2484.pedestal_left.root', x_min=-16.81, x_max=672.4, calibration_constant=3.362e-3, log_y=True, canvas_height=600)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2484.pedestal_left.root', x_min=-5e3, x_max=200e3, log_y=True, canvas_height=800)
# sort_cerenkov_event_by_adc('cerenkovana.cosmic_run_2484.pedestal_left.root', calibration_constant=3.362e-3)

# 20190312_testbeam_cerenkov_cosmic_trigger
# gStyle.SetOptStat('emr')
# plot_dndx_vs_time_of_flight()
# plot_cerenkov_pulse('V1742Analysis.run_2430_0001.root', spill=12, event=1, gate_min=250, gate_max=550)
# plot_cerenkov_pulse('V1742Analysis.run_2430_0001.root', spill=16, event=1, gate_min=250, gate_max=550)
# plot_cerenkov_pulse('V1742Analysis.run_2430_0001.root', spill=26, event=1, gate_min=250, gate_max=550)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2430.root', x_min=-16.81, x_max=100., calibration_constant=3.362e-3, log_y=True, canvas_height=600)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2430.root', x_min=-5e3, x_max=30e3, log_y=True, canvas_height=800)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2431.root', x_min=-5e3, x_max=30e3, log_y=True, canvas_height=800)
# spills = get_pulse_spill_numbers('cerenkovana.cosmic_run_2440.root', calibration_constant=3.362e-3)
# print('spills = {}'.format(spills))
# for spill in [350]:
# for spill in [289]:
# for spill in [161]:
# for spill in [350, 289, 161, 389]:
# for spill in spills:
    # plot_cerenkov_pulse('V1742Analysis.run_2440.root', spill=spill, event=1, channel=5, gate_min=200, gate_max=800, trigger_channels=[0, 2])
    # plot_cerenkov_pulse('V1742Analysis.run_2440.root', spill=spill, event=1, channel=5, gate_min=200, gate_max=800)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2440.root', x_min=-16.81, x_max=600., calibration_constant=3.362e-3, log_y=True, canvas_height=600)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2440.root', x_min=-5e3, x_max=1000.e3, log_y=True, canvas_height=600)

# 20190311_testbeam_seal_cerenkov
# gStyle.SetOptStat(0)
# compare_cerenkov_adc_spectra(['cerenkovana.cosmic_run_2388.root',
#                               'cerenkovana.cosmic_run_2395.root',
#                               'cerenkovana.cosmic_run_2414.root',
#                               'cerenkovana.cosmic_run_2423.root'],
#                              x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3,
#                              colors=[kRed, kBlue, kGreen + 2, kMagenta + 1],
#                              legend_texts=['plastic wraps (ineffective)',
#                                            'cap Swageloks',
#                                            '+ tape LED ports + Tedlar cover',
#                                            'Tedlar window'])

# 20190306_testbeam_tof_pmt_calibration
# gStyle.SetOptStat('emr')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_tof_run_2408.root', x_min=-2000, x_max=70e3)
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_tof_run_2409.root', x_min=-2000, x_max=70e3)
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_tof_run_2410.root', x_min=-2000, x_max=70e3)
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_tof_run_2411.root', x_min=-2000, x_max=100e3)
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_tof_run_2412.root', x_min=-2000, x_max=30e3)
# plot_cerenkov_pulse('V1742Analysis.run_2411_0001.root', gate_min=450, gate_max=580)
# plot_cerenkov_pulse('V1742Analysis.run_2411_0001.root', gate_min=450, gate_max=700)

# 20190226_testbeam_cerenkov_cosmic
# gStyle.SetOptStat('emr')
# gStyle.SetOptFit(1)
# gStyle.SetOptStat(0)
# compare_cerenkov_adc_spectra(['cerenkovana.cosmic_run_2388.root', 'cerenkovana.cosmic_run_2395.root', 'cerenkovana.cosmic_run_2414.root'],
#                              x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3,
#                              legend_texts=['plastic wraps (ineffective)', 'cap Swageloks', '+ tape LED ports + Tedlar cover'])
# compare_cerenkov_adc_spectra(['cerenkovana.cosmic_run_2395.root', 'cerenkovana.cosmic_run_2388.root'], x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# compare_cerenkov_adc_spectra(['cerenkovana.cosmic_run_2379.root', 'cerenkovana.cosmic_run_2388.root'], x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_1.root', x_min=10e3, x_max=70e3)
# plot_cerenkov_calibration_constant_vs_light_level()
# plot_cherenkov_index_of_refaction_air()
# plot_cerenkov_trigger_rate_vs_threshold()
# plot_cerenkov_pulse('V1742Analysis.root')
# plot_cerenkov_hit_count_per_event('cerenkovana.root')
# plot_cerenkov_adc_spectrum('cerenkovana.root')
# plot_cerenkov_adc_spectrum('cerenkovana.gate.root')
# plot_cerenkov_hit_count_per_event('cerenkovana.calibration_run_1.root')
# plot_cerenkov_hit_count_per_event('cerenkovana.calibration_run_4.root')
# plot_cerenkov_pulse('V1742Analysis.calibration_run_1.root', gate_min=151, gate_max=1024)
# plot_cerenkov_pulse('V1742Analysis.calibration_run_2.root')
# plot_cerenkov_pulse('V1742Analysis.calibration_run_3.root')
# plot_cerenkov_pulse('V1742Analysis.calibration_run_4.root')
# plot_cerenkov_pulse('V1742Analysis.calibration_run_5.root')
# plot_cerenkov_pulse('V1742Analysis.cosmic_run_1.root', gate_min=200, gate_max=600)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_1.root', x_min=-5e3, x_max=40e3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_1.root', x_min=-5e3, x_max=150e3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_1.root', x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_1.root', x_min=-16.81, x_max=504.3, calibration_constant=3.362e-3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2328.root', x_min=-5e3, x_max=40e3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2328.root', x_min=-5e3, x_max=150e3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2328.root', x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2328.root', x_min=-16.81, x_max=504.3, calibration_constant=3.362e-3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2379.root', x_min=-5e3, x_max=40e3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2379.root', x_min=-5e3, x_max=150e3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2379.root', x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2379.root', x_min=-16.81, x_max=504.3, calibration_constant=3.362e-3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2388.root', x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2388.root', x_min=-16.81, x_max=504.3, calibration_constant=3.362e-3, log_y=True)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_2395.root', x_min=-16.81, x_max=134.48, calibration_constant=3.362e-3)
# plot_cerenkov_adc_spectrum('cerenkovana.cosmic_run_1.root', adc_method='pulse')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_1.root')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_2.root')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_3.root')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_4.root')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_5.root')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_1.root', adc_method='pulse')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_2.root', adc_method='pulse')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_3.root', adc_method='pulse')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_4.root', adc_method='pulse')
# plot_cerenkov_adc_spectrum_calibration('cerenkovana.calibration_run_5.root', adc_method='pulse')

# 20190220_testbeam_sim_intro
# gStyle.SetOptStat('emr')
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_10.root', 0, 20000, bin_count=2000, normalization_factor=200, noise_particle=True)
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_10.root', 0, 20000, bin_count=2000, normalization_factor=200, noise_particle=False)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_10.root.noise_particle_True.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=10, x_min=0, x_max=20000, y_min=1.e-2, noise_particle=True)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_10.root.noise_particle_False.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, x_min=500, x_max=2000, y_max=3., noise_particle=False)
# plot_det_sim_particle_count_per_event('text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_10.root.root', x_max=10)
# filenames = [
#     'g4bl.b_-0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-0.675T.proton.64000.root.job_1_20000.799.72m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root',
#     'g4bl.b_-1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root.hist.root'
# ]
# b_fields = [
#     -0.225,
#     -0.45,
#     -0.675,
#     -0.9,
#     -1.125,
#     -1.35,
#     -1.575,
#     -1.8
# ]
# pids = [-11, -13, 211, 321, 2212]
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_negative',
#                                canvas_height=800)

# 20190215_testbeam_helium_momentum_resolution
# print_radiation_length()
# print_momentum_resolution()
# plot_momentum_resolution('air')
# plot_momentum_resolution('helium')
# plot_momentum_resolution('helium_pipe')

# 20190204_testbeam_shielding_east
# gStyle.SetOptStat(0)
# plot_noise_particle_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_9.root', show_boundary=True)
# plot_noise_particle_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_10.root', show_boundary=True)

# 20190126_testbeam_shielding_upstream
# gStyle.SetOptStat(0)
# filenames = [
#     'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.no_shielding_2.root',
#     'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_5.root',
#     'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_6.root',
#     # 'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_7.root',
#     'g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_8.root'
# ]
# print_saved_particle_count(filenames=[filename + '.hist.root' for filename in filenames])
# for filename in filenames:
#     # plot_noise_particle_root(filename, show_boundary=True)
#     # save_particle_momentum_root(filename, 0, 20000, bin_count=2000, normalization_factor=200, noise_particle=True)
#     plot_saved_particle_momentum(filename + '.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=20, x_min=0, x_max=20000, y_min=1.e-2, noise_particle=True)
#     # break
# gStyle.SetOptStat('nemr')
# compare_det_sim_particle_count_per_event(['text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.no_shielding_2.root.root',
#                                           'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_5.root.root',
#                                           'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_6.root.root',
#                                           'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_8.root.root'],
#                                          ['No Shielding', '2 Blocks', '3 Blocks (w.c. 3)', '3 Blocks (ds. col.)'],
#                                          [kBlack, kRed, kBlue, kGreen + 2],
#                                          'four')

# 20190116_testbeam_timing_structure
# gStyle.SetOptStat('emr')
# plot_particle_timing_detector_event()

# 20181213_testbeam_shielding_noise_particle
# gStyle.SetOptStat(0)
# plot_noise_particle_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding.root', show_boundary=True)
# plot_noise_particle_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root', show_boundary=True)
# plot_noise_particle_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root', show_boundary=True)
# plot_noise_particle_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_3.root', show_boundary=True)
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding.root', 0, 20000, bin_count=2000, normalization_factor=200, noise_particle=True)
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root', 0, 20000, bin_count=2000, normalization_factor=199.98, noise_particle=True)
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root', 0, 20000, bin_count=2000, normalization_factor=200, noise_particle=True)
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_3.root', 0, 20000, bin_count=2000, normalization_factor=200, noise_particle=True)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=5, x_min=0, x_max=20000, noise_particle=True)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding.root.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=5, x_min=0, x_max=20000, noise_particle=True)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=5, x_min=0, x_max=20000, noise_particle=True)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_3.root.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=5, x_min=0, x_max=20000, noise_particle=True)
# gStyle.SetOptStat('nemr')
# plot_det_sim_particle_count_per_event('text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root.root')
# plot_det_sim_particle_count_per_event('text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root.root')
# compare_det_sim_particle_count_per_event('text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root.root',
                                         # 'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root.root')
# compare_det_sim_particle_count_per_event(['text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root.root',
#                                           'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root.root'],
#                                          ['No Shielding', '2 Shielding Blocks'],
#                                          [kBlack, kRed + 1],
#                                          'two')
# compare_det_sim_particle_count_per_event(['text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_9999.199.98m.no_shielding.root.root',
#                                           'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_2.root.root',
#                                           'text_gen.g4bl.b_-0.9T.proton.64000.MergedAtstart_linebeam.trigger.root.job_1_10000.200m.shielding_3.root.root'],
#                                          ['No Shielding', '2 Shielding Blocks', '3 Shielding Blocks'],
#                                          [kBlack, kRed + 1, kBlue + 1],
#                                          'three')
# print_saved_particle_count()

# testbeam_beamline_simulation
# filenames = [
#     'g4bl.b_-0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-0.675T.proton.64000.root.job_1_20000.799.72m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root',
#     'g4bl.b_-1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root.hist.root'
# ]
# b_fields = [
#     -0.225,
#     -0.45,
#     -0.675,
#     -0.9,
#     -1.125,
#     -1.35,
#     -1.575,
#     -1.8
# ]
# pids = [-11, -13, 211, 321, 2212]
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_negative')
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_negative',
#                                y_axis_title='Good Particles per Month (1M per Spill)',
#                                scaling_factor=60 * 24 * 30)
# print_particle_count_vs_b_field(filenames=filenames,
#                                 b_fields=b_fields,
#                                 pids=pids)
# filenames = [
#     'g4bl.b_0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_0.675T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_0.9T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     # 'g4bl.b_1.35T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_1.35T.proton.64000.root.job_1_40000.1599.76m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#     'g4bl.b_1.8T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root'
# ]
# b_fields = [
#     0.225,
#     0.45,
#     0.675,
#     0.9,
#     1.125,
#     1.35,
#     1.575,
#     1.8]
# pids = [11, 13, -211, -321, -2212]
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_positive')
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_positive',
#                                y_axis_title='Good Particles per Month (1M per Spill)',
#                                scaling_factor=60 * 24 * 30)
# print_particle_count_vs_b_field(filenames=filenames,
#                                 b_fields=b_fields,
#                                 pids=pids)
# print_figure_tex()
# save_particle_momentum_root('g4bl.b_1.35T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=799.92)
# plot_saved_particle_momentum('g4bl.b_1.35T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root', b_field=1.35, beam_momentum=64, log_y=True, rebin=2, x_min=1000., x_max=2500.)
# save_particle_momentum_root('g4bl.b_1.35T.proton.64000.root.job_1_40000.1599.76m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=1599.76)
# plot_saved_particle_momentum('g4bl.b_1.35T.proton.64000.root.job_1_40000.1599.76m.kineticEnergyCut_20.root.hist.root', b_field=1.35, beam_momentum=64, log_y=True, rebin=2, x_min=1000., x_max=2500.)
# save_particle_momentum_root('g4bl.b_-0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_-0.675T.proton.64000.root.job_1_20000.799.72m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=799.72)
# save_particle_momentum_root('g4bl.b_0.675T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_-1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_-1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# plot_saved_particle_momentum('g4bl.b_-0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=-0.225, beam_momentum=64, log_y=True, rebin=2, x_min=0., x_max=1000.)
# plot_saved_particle_momentum('g4bl.b_0.225T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=0.225, beam_momentum=64, log_y=True, rebin=2, x_min=0., x_max=1000.)
# plot_saved_particle_momentum('g4bl.b_-0.675T.proton.64000.root.job_1_20000.799.72m.kineticEnergyCut_20.root.hist.root', b_field=-0.675, beam_momentum=64, log_y=True, rebin=2, x_min=250., x_max=2000.)
# plot_saved_particle_momentum('g4bl.b_0.675T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=0.675, beam_momentum=64, log_y=True, rebin=2, x_min=250., x_max=2000.)
# plot_saved_particle_momentum('g4bl.b_-1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=-1.125, beam_momentum=64, log_y=True, rebin=2, x_min=750., x_max=2500.)
# plot_saved_particle_momentum('g4bl.b_1.125T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=1.125, beam_momentum=64, log_y=True, rebin=2, x_min=750., x_max=2500.)
# plot_saved_particle_momentum('g4bl.b_-1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=-1.575, beam_momentum=64, log_y=True, rebin=2, x_min=1250., x_max=3000.)
# plot_saved_particle_momentum('g4bl.b_1.575T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=1.575, beam_momentum=64, log_y=True, rebin=2, x_min=1250., x_max=3000.)

# 20181203_testbeam_bridge_beam_detsim
# gStyle.SetOptStat(0)
# plot_beamline_sim_global_timing()
# plot_beamline_sim_spill_timing()
# plot_detsim_momentum('testbeamana.g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root')
# plot_detsim_fls_hit_gev('testbeamana.g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root')

# 20181129_electron_gun_physics
# plot_birks_law()

# 20181123_testbeam_beam_sim_high_stat
# plot_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=599.3, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# save_particle_momentum_csv('g4bl.b_-0.9T.proton.64000.root.job_1_2000.40m.kineticEnergyCut_20.csv', 0, 3000, bin_count=300, normalization_factor=40.)
# save_particle_momentum_csv('g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv', 0, 3000, bin_count=300, normalization_factor=599.3)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=2)
# save_particle_momentum_root('g4bl.b_-0.9T.proton.64000.root.job_30000_32000.40m.kineticEnergyCut_20.root', 0, 3000, bin_count=300, normalization_factor=40.)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_30000_32000.40m.kineticEnergyCut_20.root.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=2)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_1_32000.639.3m.kineticEnergyCut_20.hadd.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=2)
# split_rows('data/b_-1.35T.pnfs2xrootd.txt', 5000)
# split_rows('data/b_-1.8T.pnfs2xrootd.txt', 10000)
# split_rows('data/b_-0.45T.pnfs2xrootd.txt', 10000)
# split_rows('data/b_0.9T.pnfs2xrootd.txt', 10000)
# save_particle_momentum_root('g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 3000, bin_count=300, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 3000, bin_count=300, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=600.)
# plot_saved_particle_momentum('g4bl.b_-0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=-0.45, beam_momentum=64, log_y=True, rebin=2, x_max=1500.)
# plot_saved_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_1_30000.599.3m.kineticEnergyCut_20.csv.hist.root', b_field=-0.9, beam_momentum=64, log_y=True, rebin=2, x_min=500., x_max=2000.)
# plot_saved_particle_momentum('g4bl.b_-1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=-1.35, beam_momentum=64, log_y=True, rebin=2, x_min=1000, x_max=2500.)
# plot_saved_particle_momentum('g4bl.b_-1.8T.proton.64000.root.job_1_22500.600m.kineticEnergyCut_20.root.hist.root', b_field=-1.8, beam_momentum=64, log_y=True, rebin=2, x_min=1500., x_max=3000.)
# plot_particle_count_vs_b_field()
# print_particle_count_vs_b_field()
# plot_particle_count_vs_b_field(y_axis_title='Good Particles per Month (1M per Spill)', scaling_factor=60 * 24 * 30)
# save_particle_momentum_root('g4bl.b_0.9T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_1.8T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=799.92)
# save_particle_momentum_root('g4bl.b_0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# save_particle_momentum_root('g4bl.b_1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root', 0, 5000, bin_count=500, normalization_factor=800.)
# plot_saved_particle_momentum('g4bl.b_0.9T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=0.9, beam_momentum=64, log_y=True, rebin=2, x_min=500., x_max=2000.)
# plot_saved_particle_momentum('g4bl.b_1.8T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root', b_field=1.8, beam_momentum=64, log_y=True, rebin=2, x_min=1500., x_max=3000.)
# plot_saved_particle_momentum('g4bl.b_0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=0.45, beam_momentum=64, log_y=True, rebin=2, x_max=1500.)
# plot_saved_particle_momentum('g4bl.b_1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root', b_field=1.35, beam_momentum=64, log_y=True, rebin=2, x_min=1000., x_max=2500.)
# filenames = ['g4bl.b_0.45T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#              'g4bl.b_0.9T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#              'g4bl.b_1.35T.proton.64000.root.job_1_20000.800m.kineticEnergyCut_20.root.hist.root',
#              'g4bl.b_1.8T.proton.64000.root.job_1_20000.799.92m.kineticEnergyCut_20.root.hist.root']
# b_fields = [0.45, 0.9, 1.35, 1.8]
# pids = [11, 13, -211, -321, -2212]
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_positive')
# plot_particle_count_vs_b_field(filenames=filenames,
#                                b_fields=b_fields,
#                                pids=pids,
#                                suffix='.b_positive',
#                                y_axis_title='Good Particles per Month (1M per Spill)',
#                                scaling_factor=60 * 24 * 30)
# print_particle_count_vs_b_field(filenames=filenames,
#                                 b_fields=b_fields,
#                                 pids=pids)

# 20181115_testbeam_proton_secondary_beam
# gStyle.SetOptStat(0)
# plot_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_1_2000.40m.kineticEnergyCut_20.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=40., y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.proton.64000.root.job_1_6000.119.3m.kineticEnergyCut_20.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=119.3, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_b_field()

# 20181105_testbeam_g4bl_speed
# plot_particle_momentum('g4bl.b_0.9T.pi+.64000.root.job_1_6000.80m.kineticEnergyCut_20.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=80., y_title='Particle Count per 1M Beam Particles', b_field=0.9, beam_momentum=64)
# print('Charge() = {}'.format(PDG.GetParticle('e+').Charge()))
# use kineticEnergyCut=20 only
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.301.4m.kineticEnergyCut_20.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=301.4, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.87.6m.kineticEnergyCut_20.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=87.6, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# compute_minimum_kinetic_energy()
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_6000.53.63m.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=53.63, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.1537m.kill.keep.kineticEnergyCut.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=1537, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.1998m.kill.keep.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=1998, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.399.4m.keep.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=399.4, y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# no normalization
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_6000.53.63m.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.399.4m.keep.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.1998m.kill.keep.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_2000.1537m.kill.keep.kineticEnergyCut.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=1., y_title='Particle Count per 1M Beam Particles', b_field=-0.9, beam_momentum=64)

# 20181031_beamline_sim_update
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.1_100.610k.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=0.61, y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('g4bl.b_-0.9T.pi+.64000.root.job_1_4000.33.66m.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=33.66, y_title='Particle Count per 1M Beam Particles')
# compute_minimum_kinetic_energy()

# 20181025_testbeam_trigger_rate
# gStyle.SetOptStat(0)
# save_trigger_rate()
# plot_trigger_rate(plot_rate=False)
# plot_trigger_rate(plot_rate=True)

# 20180912_testbeam_radiation_collimator
# save_momentum_collimator_up()
# plot_momentum_collimator_up()
# print_momentum_collimator_up()

# 20180910_testbeam_cherenkov_length
# plot_energy_loss_vs_cherenkov_length()

# 20180904_testbeam_readout_sim
# gStyle.SetOptStat('emr')
# plot_dcs_threshold()

# 20180731_doe
# plot_cherenkov_index_of_refaction()
# plot_time_of_flight_diff(distance=14.8, y_max=3.e6, canvas_height=600)

# 20180726_testbeam_detsim_config
# gStyle.SetOptStat(0)
# plot_birks_law()
# plot_cherenkov_photon_count()
# plot_dt_dz_collection_rate()
# plot_fiber_brightness()
# plot_fiber_attenuation()

# 20180625_testbeam_64_32_16_8GeV_different_bs
# print_radiation_length()
# plot_particle_count_vs_secondary_beam_energy('gr_total', y_min=0, y_max=30)
# plot_particle_count_vs_secondary_beam_energy('gr_pi', y_min=0, y_max=15)
# plot_particle_count_vs_secondary_beam_energy('gr_proton', y_min=0, y_max=15)
# plot_particle_count_vs_secondary_beam_energy('gr_k', y_min=0, y_max=1.8)
# plot_particle_count_vs_secondary_beam_energy('gr_e', y_min=0, y_max=3.)
# plot_particle_count_vs_secondary_beam_energy('gr_mu', y_min=0, y_max=1.8)
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
# plot_particle_momentum('beamline.py.in.job_1_1800.18m.b_-0.9T.pi+_64gev.root.csv', 700, 1800, title='64 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=18., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beamline.py.in.job_1_1800.27m.b_-0.9T.pi+_32gev.root.csv', 700, 1800, title='32 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=27., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beamline.py.in.job_1_900.45m.b_-0.9T.pi+_16gev.root.csv', 700, 1800, title='16 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=45., y_title='Particle Count per 1M Beam Particles')
# plot_particle_momentum('beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root.csv', 700, 1800, title='8 GeV Secondary Beam', y_max=0., bin_count=11, y_title_offset=1.4, normalization_factor=90., y_title='Particle Count per 1M Beam Particles')
# get_particle_count_vs_secondary_beam_energy(suffix='b_-1.8T', csv_64gev='g4bl.b_-1.8T.pi+.64000.csv', norm_64gev=4., csv_32gev='g4bl.b_-1.8T.pi+.32000.csv', norm_32gev=12., csv_16gev='g4bl.b_-1.8T.pi+.16000.csv', norm_16gev=24.32, csv_8gev='g4bl.b_-1.8T.pi+.8000.csv', norm_8gev=103.8,)
# get_particle_count_vs_secondary_beam_energy(suffix='b_-1.35T', csv_64gev='g4bl.b_-1.35T.pi+.64000.csv', norm_64gev=4., csv_32gev='g4bl.b_-1.35T.pi+.32000.csv', norm_32gev=12., csv_16gev='g4bl.b_-1.35T.pi+.16000.csv', norm_16gev=28.76, csv_8gev='g4bl.b_-1.35T.pi+.8000.csv', norm_8gev=103.7,)
# get_particle_count_vs_secondary_beam_energy(suffix='b_-0.45T', csv_64gev='g4bl.b_-0.45T.pi+.64000.csv', norm_64gev=4., csv_32gev='g4bl.b_-0.45T.pi+.32000.csv', norm_32gev=12., csv_16gev='g4bl.b_-0.45T.pi+.16000.csv', norm_16gev=24.5, csv_8gev='g4bl.b_-0.45T.pi+.8000.csv', norm_8gev=103.75,)
# get_particle_count_vs_secondary_beam_energy(suffix='b_-0.9T', csv_64gev='beamline.py.in.job_1_1800.18m.b_-0.9T.pi+_64gev.root.csv', norm_64gev=18., csv_32gev='beamline.py.in.job_1_1800.27m.b_-0.9T.pi+_32gev.root.csv', norm_32gev=27., csv_16gev='beamline.py.in.job_1_900.45m.b_-0.9T.pi+_16gev.root.csv', norm_16gev=45., csv_8gev='beamline.py.in.job_1_900.90m.b_-0.9T.pi+_8gev.root.csv', norm_8gev=90.,)
# plot_p_vs_angle_16_degree()

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
