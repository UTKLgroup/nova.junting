import random
from rootalias import *


subspillnumber = 1
subspillcount = 1
jobsinspill = subspillnumber%subspillcount
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
    # exit loop when we finally get the right range
    return offset


figure_dir = '/Users/juntinghuang/beamer/20171211_test_beam_geometry/figures'
h1 = TH1D('h1', 'h1', 600, -1, 5)
for i in range(300000):
    h1.Fill(RandomOffsetSeconds())

c1 = TCanvas('c1', 'c1', 800, 600)
# gStyle.SetOptStat('emr')
gStyle.SetOptStat(0)
set_margin()
set_h1_style(h1)
h1.GetXaxis().SetTitle('Time (s)')
h1.GetYaxis().SetTitle('Pion Count')

h1.Draw()

c1.Update()
c1.SaveAs('{}/beam_time.pdf'.format(figure_dir))
input('Press any key to continue.')
