from rootalias import *
from pprint import pprint
import csv
from math import pi, cos, sin, atan
import numpy as np


PDG = TDatabasePDG()
SPEED_OF_LIGHT = 3.e8              # m/s
ELEMENTARY_CHARGE = 1.60217662e-19 # coulomb
INCH_TO_METER = 2.54 / 100.
DEGREE_TO_RADIAN = 3.14 / 180.
FIGURE_DIR = '/Users/juntinghuang/beamer/20180211_testbeam_high_stat/figures'
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


def plot_time_of_flight():
    distance = 6.075            # m
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

    c1 = TCanvas('c1', 'c1', 800, 800)
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
    grs[0].GetYaxis().SetRangeUser(9.9e3, 2.e5)
    grs[0].GetXaxis().SetRangeUser(1.e-1, 3.)
    grs[0].GetYaxis().SetTitleOffset(1.8)
    lg1.AddEntry(grs[0], names[0], 'l')

    for i in range(1, len(names)):
        set_graph_style(grs[i])
        grs[i].Draw('sames,L')
        lg1.AddEntry(grs[i], names[i], 'l')
    lg1.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_time_of_flight.pdf'.format(figure_dir))
    input('Press any key to continue.')


def plot_time_of_flight_diff():
    distance = 6.075            # m
    # distance = 9.1            # m
    # distance = 12.            # m
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

    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    gPad.SetLogy()
    gPad.SetGrid()

    lg1 = TLegend(0.5, 0.6, 0.85, 0.9)
    set_legend_style(lg1)

    set_graph_style(grs[0])
    grs[0].Draw('AL')
    grs[0].GetXaxis().SetTitle('Momentum (GeV)')
    grs[0].GetYaxis().SetTitle('Time of Flight (ps)')
    grs[0].GetYaxis().SetRangeUser(10, 1.e6)
    grs[0].GetXaxis().SetRangeUser(0., 3.)
    grs[0].GetYaxis().SetTitleOffset(1.8)
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
    c1.SaveAs('{}/plot_time_of_flight_diff.pdf'.format(figure_dir))
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
    keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
    for key in keys:
        print('key = {}'.format(key))
        track_count = 0
        for track in tf1.Get(key):
            track_count += 1
            pass_all = track.TrackPresentstart_line and \
                       track.TrackPresenttof_upstream and \
                       track.TrackPresentwire_chamber_1_detector and \
                       track.TrackPresentwire_chamber_2_detector and \
                       track.TrackPresentwire_chamber_3_detector and \
                       track.TrackPresentwire_chamber_4_detector and \
                       track.TrackPresenttof_downstream

            if track_count % 100000 == 0:
                print('track_count = {}'.format(track_count))

            if pass_all:
                print('passed!')
                particle = [track.EventID, track.TrackID, track.TrackPresenttof_downstream, track.xtof_downstream, track.ytof_downstream, track.ztof_downstream, track.ttof_downstream, track.Pxtof_downstream, track.Pytof_downstream, track.Pztof_downstream, track.PDGidtof_downstream, track.ParentIDtof_downstream]
                particles.append(particle)

                pid = track.PDGidtof_downstream
                momentum = (track.Pxtof_downstream**2 + track.Pytof_downstream**2 + track.Pztof_downstream**2)**0.5
                if pid not in pid_momentums:
                    pid_momentums[pid] = [momentum]
                else:
                    pid_momentums[pid].append(momentum)
                print('track.PDGidtof_downstream = {}'.format(track.PDGidtof_downstream))
                print('momentum = {}'.format(momentum))

    with open('{}/{}.csv'.format(DATA_DIR, filename), 'w') as f_fraction:
        for particle in particles:
            particle = list(map(str, particle))
            f_fraction.write('{}\n'.format(','.join(particle)))

    pprint(pid_momentums)


def plot_particle_momentum(filename, x_min, x_max, y_max):
    pid_momentums = {}
    with open('{}/{}'.format(DATA_DIR, filename)) as f_csv:
        for row in csv.reader(f_csv, delimiter=','):
            pid = int(float(row[-2]))
            px = float(row[-5])
            py = float(row[-4])
            pz = float(row[-3])
            momentum = (px**2 + py**2 + pz**2)**0.5

            if pid not in pid_momentums:
                pid_momentums[pid] = [momentum]
            else:
                pid_momentums[pid].append(momentum)

    pid_hists = {}
    h_all = TH1D('h_all', 'h_all', 50, x_min, x_max)
    for pid, momentums in pid_momentums.items():
        hist = TH1D('h_{}'.format(pid), 'h_{}'.format(pid), 50, x_min, x_max)
        for momentum in momentums:
            hist.Fill(momentum)
            h_all.Fill(momentum)
        pid_hists[pid] = hist

    c1 = TCanvas('c1', 'c1', 800, 800)
    set_margin()
    gStyle.SetOptStat(0)

    pids = pid_hists.keys()
    colors = [
        kBlack,
        kRed + 2,
        kBlue + 2,
        kGreen + 2,
        kMagenta + 2
    ]

    # h_stack = THStack('h_stack', 'h_stack')
    lg1 = TLegend(0.575, 0.55, 0.84, 0.84)
    set_legend_style(lg1)

    pids = sorted(pids)
    for i, pid in enumerate(pids):
        hist = pid_hists[pid]
        set_h1_style(hist)
        hist.SetLineColor(colors[i])
        # h_stack.Add(hist)
        # hist.SetFillColor(colors[i])

        lg1.AddEntry(hist, '{} ({:.0f})'.format(PDG.GetParticle(pid).GetName(), hist.GetEntries()), 'l')

        if i == 0:
            hist.Draw()
            hist.GetXaxis().SetTitle('Momentum (MeV)')
            hist.GetYaxis().SetTitle('Particle Count')
            hist.GetYaxis().SetRangeUser(0, y_max)
        else:
            hist.Draw('sames')

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(43)
    latex.SetTextSize(28)
    latex.DrawLatex(0.59, 0.44, 'mean = {:.0f} MeV'.format(h_all.GetMean()))
    latex.DrawLatex(0.59, 0.5, 'total entry = {:.0f}'.format(h_all.GetEntries()))

    lg1.Draw()
    # set_h1_style(h_stack)
    # h_stack.Draw()

    c1.Update()
    c1.SaveAs('{}/plot_particle_momentum.{}.pdf'.format(FIGURE_DIR, filename))
    input('Press any key to continue.')



# 20180211_testbeam_high_stat
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_0.45T.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_1.8T.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.root')
# save_particle_to_csv('beam.py.in.10_spill.job_1_300.10k_per_job.b_-1.8T.root')
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_0.45T.root.csv', 300, 2000, 20)
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_1.8T.root.csv', 2000, 6000, 10)
# plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_-0.45T.root.csv', 300, 2000, 22)
plot_particle_momentum('beam.py.in.10_spill.job_1_300.10k_per_job.b_-1.8T.root.csv', 2000, 6000, 10)

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
