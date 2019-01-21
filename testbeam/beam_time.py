import random
from rootalias import *


subspillnumber = 1              # job number: 1, 2, ..., 30, 31, 31, ...
subspillcount = 30              # number of jobs for a spill, e.g. for a spill of 300k events,
                                # if i use 10k events per job, the number of jobs here is 30
jobsinspill = (subspillnumber - 1) % subspillcount
BatchesPerOrbit = 7
BucketsPerBatch = 84
BucketsPerOrbit = BatchesPerOrbit * BucketsPerBatch

bucketcenterspacing = float("18.8e-9")
bucketwidth = float("2.2e-9")
batchlength = bucketcenterspacing * float(BucketsPerBatch)
orbitlength = batchlength * float(BatchesPerOrbit)
spillduration = 4.2
OrbitsInSpill = spillduration / orbitlength
filledbatches = (1,2,3,4,5,6) # (out of BatchesPerOrbit)

print('orbitlength = {}'.format(orbitlength))
print('OrbitsInSpill = {}'.format(OrbitsInSpill))

# Function to return a time during the spill, weighted to get the
# time structure of the Fermilab Test Beam Facility's beam
def RandomOffsetSeconds ():
    subspillduration = spillduration/subspillcount
    subspilltimewindow_early = jobsinspill * subspillduration
    subspilltimewindow_late = subspilltimewindow_early + subspillduration

    offset = -1
    while offset < subspilltimewindow_early or offset >= subspilltimewindow_late:
        BucketInBatch = random.randint(1,BucketsPerBatch-1)
        BatchInOrbit = random.choice(filledbatches)
        OrbitInSpill = random.randint(0,int(OrbitsInSpill))

        offset = random.gauss(0,bucketwidth)
        offset += bucketcenterspacing * float(BucketInBatch)
        offset += batchlength * BatchInOrbit
        offset += orbitlength * OrbitInSpill

    return offset


def get_offset_bucket():
    BucketInBatch = random.randint(1, BucketsPerBatch - 1)
    offset = random.gauss(0, bucketwidth)
    offset += bucketcenterspacing * float(BucketInBatch)
    return offset * 1.e9

# figure_dir = '/Users/juntinghuang/beamer/20171211_test_beam_geometry/figures'
# h1 = TH1D('h1', 'h1', 600, -1, 5)
figure_dir = '/Users/juntinghuang/beamer/20190116_testbeam_shielding_upstream/figures'
h1 = TH1D('h1', 'h1', 1880, 0, 188)
total_count = 3000000
for i in range(total_count):
    if i % 1e4 == 0:
        print('i = {} / {}'.format(i, total_count))
    # h1.Fill(RandomOffsetSeconds())
    h1.Fill(get_offset_bucket())

c1 = TCanvas('c1', 'c1', 800, 600)
gStyle.SetOptStat(0)
set_margin()
set_h1_style(h1)
h1.GetXaxis().SetTitle('Time (ns)')
h1.GetYaxis().SetTitle('Proton Count')
h1.Draw()

c1.Update()
c1.SaveAs('{}/beam_time.pdf'.format(figure_dir))
input('Press any key to continue.')
