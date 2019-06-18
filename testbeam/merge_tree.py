# Author: Jason St. John, Junting Huang

import ROOT
from ROOT import gDirectory
import sys
import argparse
import os
import random
from pprint import pprint
from array import array


class MergeTree:
    BATCH_PER_ORBIT = 7.
    BUCKET_PER_BATCH = 84.
    BUCKET_WIDTH = 2.2e-9
    BUCKET_CENTER_SPACING = 18.8e-9
    SPILL_DURATION = 4.2
    FILLED_BATCHES = (1, 2, 3, 4, 5, 6)
    DEFAULT_VALUE = -123456789

    def __init__(self,
                 input_filename,
                 output_filename,
                 starter_tree,
                 max_spill,
                 spill_size,
                 spill_interval,
                 subspill_number,
                 subspill_count,
                 gamma_cutoff):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.starter_tree = starter_tree
        self.max_spill = max_spill
        self.spill_size = spill_size
        self.spill_interval = spill_interval
        self.subspill_number = subspill_number
        self.subspill_count = subspill_count
        self.gamma_cutoff = gamma_cutoff
        self.jobs_in_spill = self.subspill_number % subspill_count

        self.batch_length = MergeTree.BUCKET_CENTER_SPACING * MergeTree.BUCKET_PER_BATCH
        self.orbit_length = self.batch_length * MergeTree.BATCH_PER_ORBIT
        self.orbit_in_spill = MergeTree.SPILL_DURATION / self.orbit_length

        self.variables = ['x', 'y', 'z', 't', 'Px', 'Py', 'Pz', 'PDGid', 'Edep', 'ParentID', 'EventID', 'TrackID']
        self.category_detectors = {
            # 'start_line': ['start_line'],
            'wire_chambers': ['wire_chamber_1_detector', 'wire_chamber_2_detector', 'wire_chamber_3_detector', 'wire_chamber_4_detector'],
            'scintillator_detectors': ['tof_us', 'tof_ds', 'tof_ds_sipm', 'cherenkov', 'nova']
        }

        self.detector_categories = dict()
        for category, detectors in self.category_detectors.items():
            for detector in detectors:
                self.detector_categories[detector] = category

        self.all_detectors = []
        for detectors in self.category_detectors.values():
            self.all_detectors += detectors

        ROOT.gROOT.SetBatch(0)

    def random_offset_seconds(self):
        subspill_duration = MergeTree.SPILL_DURATION / self.subspill_count
        subspill_time_window_early = self.jobs_in_spill * subspill_duration
        subspill_time_window_late = subspill_time_window_early + subspill_duration
        offset = -1.
        while offset < subspill_time_window_early or offset >= subspill_time_window_late:
            bucket_in_batch = random.randint(1, MergeTree.BUCKET_PER_BATCH - 1)
            batch_in_orbit = random.choice(MergeTree.FILLED_BATCHES)
            orbit_in_spill = random.randint(0, int(self.orbit_in_spill))
            offset = random.gauss(0, MergeTree.BUCKET_WIDTH)
            offset += MergeTree.BUCKET_CENTER_SPACING * bucket_in_batch
            offset += self.batch_length * batch_in_orbit
            offset += self.orbit_length * orbit_in_spill

        return offset

    def add_tree(self, spill_trees, spill, pointers):
        tree_name = 'EventTree_Spill' + str(spill)
        spill_trees[spill] = ROOT.TTree(tree_name, tree_name)

        pointers[spill, 'SpillID'] = array('i', [0])
        pointers[spill, 'EventID'] = array('i', [0])
        pointers[spill, 'TrackID'] = array('i', [0])

        spill_trees[spill].Branch('SpillID', pointers[spill, 'SpillID'], 'SpillID/I')
        spill_trees[spill].Branch('EventID', pointers[spill, 'EventID'], 'EventID/I')
        spill_trees[spill].Branch('TrackID', pointers[spill, 'TrackID'], 'TrackID/I')

        for detector in self.all_detectors:
            name = 'TrackPresent' + detector
            pointers[spill, name] = array('B', [0])
            spill_trees[spill].Branch(name, pointers[spill, name], name + '/O')
            for variable in self.variables:
                if variable == 'EventID' or variable == 'TrackID':
                    continue
                name = variable + detector
                pointers[spill, name] = array('d', [0])
                spill_trees[spill].Branch(name, pointers[spill, name], name + '/d')

    def run(self):
        if not os.path.isfile(self.input_filename):
            print("Cannot find %s.  Exiting.".format(self.input_filename))
            sys.exit()
        tfile_input = ROOT.TFile(self.input_filename)

        tfile_input.cd()
        # gDirectory.cd('VirtualDetector')
        gDirectory.cd('Detector')
        keys = [key.GetName() for key in gDirectory.GetListOfKeys()]

        detector_events = {}
        for key in gDirectory.GetListOfKeys():
            detector_events[key.GetName()] = key.ReadObj()

        for detector, events in detector_events.items():
            events.BuildIndex('EventID', 'TrackID')

        # ROOT prevents building a single tree that is too large.
        for category, detectors in self.category_detectors.items():
            line_to_process = 'struct ' + category + ' {Int_t EventID; Int_t TrackID; '
            for variable in self.variables:
                data_type = 'Float_t'
                if variable == 'EventID' or variable == 'TrackID':
                    data_type = 'Int_t'
                for detector in detectors:
                    line_to_process += data_type + ' ' + variable + detector + '; '
            line_to_process += '};'
            ROOT.gROOT.ProcessLine(line_to_process)

        category_structs = {}
        for category in self.category_detectors.keys():
            category_structs[category] = eval('ROOT.' + category + '()')
        
        for category, detectors in self.category_detectors.items():
            for detector in detectors:
                for variable in self.variables:
                    events = detector_events[detector]
                    events.SetBranchAddress(variable, ROOT.AddressOf(category_structs[category], variable + detector))

        tfile_output = ROOT.TFile(self.output_filename, 'RECREATE')
        spill_trees = {}
        pointers = {}
        track_count = 0
        spill_count = 0

        for event in detector_events[self.starter_tree]:
            track_count += 1
            (event_id, track_id) = (int(event.EventID), int(event.TrackID))
            spill = 1 + (event_id // self.spill_size)

            if self.gamma_cutoff > 0. and event.PDGid == 22:
                energy = (event.Px**2 + event.Py**2 + event.Pz**2)**0.5
                if energy < self.gamma_cutoff:
                    continue
        
            if spill not in spill_trees:
                spill_count += 1
                if spill_count > self.max_spill > 0:
                    break
                self.add_tree(spill_trees, spill, pointers)

            pointers[spill, 'SpillID'][0] = spill
            pointers[spill, 'EventID'][0] = event_id
            pointers[spill, 'TrackID'][0] = track_id

            for detector, events in detector_events.items():
                entry_number = events.GetEntryWithIndex(event_id, track_id)
                track_present = not (entry_number == -1)
                pointers[spill, 'TrackPresent' + detector][0] = track_present

                for variable in self.variables:
                    if variable == 'EventID' or variable == 'TrackID':
                        continue

                    variable_detector = variable + detector
                    value = MergeTree.DEFAULT_VALUE
                    if track_present:
                        value = getattr(category_structs[self.detector_categories[detector]], variable_detector)
                        if variable == 't':
                            random.seed(event_id)
                            value = value * 1.e-9 + self.spill_interval * float(spill) + self.random_offset_seconds()

                    pointers[spill, variable_detector][0] = value

            spill_trees[spill].Fill()

        print('{} total tracks in {}'.format(track_count, self.starter_tree))
        print('{} total spills'.format(spill_count - 1))

        tfile_output.cd()
        for tree in spill_trees.values():
            tree.Write()
        tfile_output.Close()
        tfile_input.Close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', default='beam.root',
                        help='Input filename.')
    parser.add_argument('-o', '--output_filename', default='merge_tree.root',
                        help='Output filename.')
    parser.add_argument('-t', '--starter_tree', default='nova',
                        help='The one TTree whose tracks will be iterated over. Effectively requires tracks present here.')
    parser.add_argument('--max_spill', type=int, default=-1,
                        help='Abbreviate processing to this many spills.')
    parser.add_argument('--spill_size', type=int, default=300000,
                        help='The number of G4BL events (particles launched at the target) per spill.')
    parser.add_argument('--spill_interval', type=float, default=60.,
                        help='The duration of a subrun in seconds')
    parser.add_argument('--subspill_number', type=int, default=1,
                        help='If processing the spill in parallel tranches, which subspill number is this.')
    parser.add_argument('--subspill_count', type=int, default=1,
                        help='If processing the spill in parallel tranches, how many subspills to assume.')
    parser.add_argument('--gamma_cutoff', type=float, default=0.5,
                        help='In the starterTree gammas less than this energy will be ignored. The default value is 0.5 (MeV)')

    args = parser.parse_args()
    merge_tree = MergeTree(args.input_filename,
                           args.output_filename,
                           args.starter_tree,
                           args.max_spill,
                           args.spill_size,
                           args.spill_interval,
                           args.subspill_number,
                           args.subspill_count,
                           args.gamma_cutoff)
    merge_tree.run()
