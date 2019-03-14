from ROOT import TFile, gDirectory, gROOT, TTree
import argparse
import os
from pprint import pprint
from array import array

gROOT.ProcessLine(
    'struct Particle { \
    Int_t is_noise; \
    Int_t event_id; \
    Int_t track_id; \
    Float_t x; \
    Float_t y; \
    Float_t z; \
    Float_t t; \
    Float_t px; \
    Float_t py; \
    Float_t pz; \
    Float_t pdg_id; \
    Float_t parent_id; \

    Int_t spill_number; \

    Bool_t present_tof_us; \
    Float_t x_tof_us; \
    Float_t y_tof_us; \
    Float_t z_tof_us; \
    Float_t t_tof_us; \
    Float_t px_tof_us; \
    Float_t py_tof_us; \
    Float_t pz_tof_us; \

    Bool_t present_wire_chamber_1; \
    Float_t x_wire_chamber_1; \
    Float_t y_wire_chamber_1; \
    Float_t z_wire_chamber_1; \
    Float_t t_wire_chamber_1; \
    Float_t px_wire_chamber_1; \
    Float_t py_wire_chamber_1; \
    Float_t pz_wire_chamber_1; \

    Bool_t present_wire_chamber_2; \
    Float_t x_wire_chamber_2; \
    Float_t y_wire_chamber_2; \
    Float_t z_wire_chamber_2; \
    Float_t t_wire_chamber_2; \
    Float_t px_wire_chamber_2; \
    Float_t py_wire_chamber_2; \
    Float_t pz_wire_chamber_2; \

    Bool_t present_wire_chamber_3; \
    Float_t x_wire_chamber_3; \
    Float_t y_wire_chamber_3; \
    Float_t z_wire_chamber_3; \
    Float_t t_wire_chamber_3; \
    Float_t px_wire_chamber_3; \
    Float_t py_wire_chamber_3; \
    Float_t pz_wire_chamber_3; \

    Bool_t present_wire_chamber_4; \
    Float_t x_wire_chamber_4; \
    Float_t y_wire_chamber_4; \
    Float_t z_wire_chamber_4; \
    Float_t t_wire_chamber_4; \
    Float_t px_wire_chamber_4; \
    Float_t py_wire_chamber_4; \
    Float_t pz_wire_chamber_4; \

    Bool_t present_cherenkov; \
    Float_t x_cherenkov; \
    Float_t y_cherenkov; \
    Float_t z_cherenkov; \
    Float_t t_cherenkov; \
    Float_t px_cherenkov; \
    Float_t py_cherenkov; \
    Float_t pz_cherenkov; \

    Bool_t present_tof_ds; \
    Float_t x_tof_ds; \
    Float_t y_tof_ds; \
    Float_t z_tof_ds; \
    Float_t t_tof_ds; \
    Float_t px_tof_ds; \
    Float_t py_tof_ds; \
    Float_t pz_tof_ds; \
    };' );
from ROOT import Particle


def save_particle_to_root(filename):
    filename_base, filename_ext = os.path.splitext(filename)
    tf_out = TFile('{}.trigger{}'.format(filename_base, filename_ext), 'RECREATE')

    tree = TTree('tree', 'tree')
    particle = Particle()
    tree.Branch('particle', particle, 'is_noise/I:event_id:track_id:t_tof_us/F:t_tof_ds:x:y:z:t:px:py:pz:pdg_id:parent_id')

    tf_in = TFile(filename)
    pid_momentums = {}
    particles = []
    noise_particles = []

    keys = [key.GetName() for key in gDirectory.GetListOfKeys()]
    for key in keys:
        print('key = {}'.format(key))
        track_count = 0
        for track in tf_in.Get(key):
            track_count += 1
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
                if pass_all:
                    particle.is_noise = 0
                else:
                    particle.is_noise = 1
                particle.event_id = track.EventID
                particle.track_id = track.TrackID
                particle.x = track.xnova
                particle.y = track.ynova
                particle.z = track.znova
                particle.t = track.tnova
                particle.px = track.Pxnova
                particle.py = track.Pynova
                particle.pz = track.Pznova
                particle.pdg_id = track.PDGidnova
                particle.parent_id = track.ParentIDnova

                particle.spill_number = track.SpillID

                particle.present_tof_us = track.TrackPresenttof_us
                particle.x_tof_us = track.xtof_us
                particle.y_tof_us = track.ytof_us
                particle.z_tof_us = track.ztof_us
                particle.t_tof_us = track.ttof_us
                particle.px_tof_us = track.Pxtof_us
                particle.py_tof_us = track.Pytof_us
                particle.pz_tof_us = track.Pztof_us

                particle.present_wire_chamber_1_detector = track.TrackPresentwire_chamber_1_detector
                particle.x_wire_chamber_1_detector = track.xwire_chamber_1_detector
                particle.y_wire_chamber_1_detector = track.ywire_chamber_1_detector
                particle.z_wire_chamber_1_detector = track.zwire_chamber_1_detector
                particle.t_wire_chamber_1_detector = track.twire_chamber_1_detector
                particle.px_wire_chamber_1_detector = track.Pxwire_chamber_1_detector
                particle.py_wire_chamber_1_detector = track.Pywire_chamber_1_detector
                particle.pz_wire_chamber_1_detector = track.Pzwire_chamber_1_detector

                particle.present_wire_chamber_2_detector = track.TrackPresentwire_chamber_2_detector
                particle.x_wire_chamber_2_detector = track.xwire_chamber_2_detector
                particle.y_wire_chamber_2_detector = track.ywire_chamber_2_detector
                particle.z_wire_chamber_2_detector = track.zwire_chamber_2_detector
                particle.t_wire_chamber_2_detector = track.twire_chamber_2_detector
                particle.px_wire_chamber_2_detector = track.Pxwire_chamber_2_detector
                particle.py_wire_chamber_2_detector = track.Pywire_chamber_2_detector
                particle.pz_wire_chamber_2_detector = track.Pzwire_chamber_2_detector

                particle.present_wire_chamber_3_detector = track.TrackPresentwire_chamber_3_detector
                particle.x_wire_chamber_3_detector = track.xwire_chamber_3_detector
                particle.y_wire_chamber_3_detector = track.ywire_chamber_3_detector
                particle.z_wire_chamber_3_detector = track.zwire_chamber_3_detector
                particle.t_wire_chamber_3_detector = track.twire_chamber_3_detector
                particle.px_wire_chamber_3_detector = track.Pxwire_chamber_3_detector
                particle.py_wire_chamber_3_detector = track.Pywire_chamber_3_detector
                particle.pz_wire_chamber_3_detector = track.Pzwire_chamber_3_detector

                particle.present_wire_chamber_4_detector = track.TrackPresentwire_chamber_4_detector
                particle.x_wire_chamber_4_detector = track.xwire_chamber_4_detector
                particle.y_wire_chamber_4_detector = track.ywire_chamber_4_detector
                particle.z_wire_chamber_4_detector = track.zwire_chamber_4_detector
                particle.t_wire_chamber_4_detector = track.twire_chamber_4_detector
                particle.px_wire_chamber_4_detector = track.Pxwire_chamber_4_detector
                particle.py_wire_chamber_4_detector = track.Pywire_chamber_4_detector
                particle.pz_wire_chamber_4_detector = track.Pzwire_chamber_4_detector

                particle.present_cherenkov = track.TrackPresentcherenkov
                particle.x_cherenkov = track.xcherenkov
                particle.y_cherenkov = track.ycherenkov
                particle.z_cherenkov = track.zcherenkov
                particle.t_cherenkov = track.tcherenkov
                particle.px_cherenkov = track.Pxcherenkov
                particle.py_cherenkov = track.Pycherenkov
                particle.pz_cherenkov = track.Pzcherenkov

                particle.present_tof_ds = track.TrackPresenttof_ds
                particle.x_tof_ds = track.xtof_ds
                particle.y_tof_ds = track.ytof_ds
                particle.z_tof_ds = track.ztof_ds
                particle.t_tof_ds = track.ttof_ds
                particle.px_tof_ds = track.Pxtof_ds
                particle.py_tof_ds = track.Pytof_ds
                particle.pz_tof_ds = track.Pztof_ds

                tree.Fill()
    tf_in.Close()

    tf_out.cd()
    tree.Write()
    tf_out.Close()


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    print('processing {}'.format(args.filename))
    save_particle_to_root(args.filename)
