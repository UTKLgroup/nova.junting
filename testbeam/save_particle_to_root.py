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
    Float_t t_tof_us; \
    Float_t t_tof_ds; \
    Float_t x; \
    Float_t y; \
    Float_t z; \
    Float_t t; \
    Float_t px; \
    Float_t py; \
    Float_t pz; \
    Float_t pdg_id; \
    Float_t parent_id; \
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
                particle.t_tof_us = track.ttof_us
                particle.t_tof_ds = track.ttof_ds
                particle.x = track.xnova
                particle.y = track.ynova
                particle.z = track.znova
                particle.t = track.tnova
                particle.px = track.Pxnova
                particle.py = track.Pynova
                particle.pz = track.Pznova
                particle.pdg_id = track.PDGidnova
                particle.parent_id = track.ParentIDnova
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
