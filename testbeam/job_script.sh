
function setup_nova_grid {
  source /cvmfs/nova.opensciencegrid.org/novasoft/slf6/novasoft/setup/setup_nova.sh "$@"
}

setup_nova_grid -r S18-01-19 -6 /cvmfs/nova.opensciencegrid.org/novasoft/slf6/novasoft -e /cvmfs/nova.opensciencegrid.org/externals -b maxopt
source /nova/app/users/junting/g4beamline/G4beamline-3.04/bin/g4bl-setup.sh

export G4BL_DIR=/nova/app/users/junting/g4beamline/G4beamline-3.04
export G4ABLADATA=/nova/app/users/junting/g4beamline/Geant4Data/G4ABLA3.0
export G4LEDATA=/nova/app/users/junting/g4beamline/Geant4Data/G4EMLOW6.50
export G4ENSDFSTATEDATA=/nova/app/users/junting/g4beamline/Geant4Data/G4ENSDFSTATE2.1
export G4NEUTRONHPDATA=/nova/app/users/junting/g4beamline/Geant4Data/G4NDL4.5
export G4NEUTRONXSDATA=/nova/app/users/junting/g4beamline/Geant4Data/G4NEUTRONXS1.4
export G4PIIDATA=/nova/app/users/junting/g4beamline/Geant4Data/G4PII1.3
export G4SAIDXSDATA=/nova/app/users/junting/g4beamline/Geant4Data/G4SAIDDATA1.1
export G4LEVELGAMMADATA=/nova/app/users/junting/g4beamline/Geant4Data/PhotonEvaporation4.3
export G4RADIOACTIVEDATA=/nova/app/users/junting/g4beamline/Geant4Data/RadioactiveDecay5.1
export G4REALSURFACEDATA=/nova/app/users/junting/g4beamline/Geant4Data/RealSurface1.0

PROCESS_START=300
# B_FIELD=b_1.8T
# B_FIELD=b_-1.8T
# B_FIELD=b_0.45T
# B_FIELD=b_-0.45T
# B_FIELD=b_-0.9T
# B_FIELD=b_-1.35T
B_FIELD=b_-1.8T

ifdh cp /pnfs/nova/persistent/users/junting/testbeam/merge_tree.py ./merge_tree.py
ifdh cp /pnfs/nova/persistent/users/junting/testbeam/beam.py.${B_FIELD}.in ./beam.py.in

PARTICLE=proton
MOMENTUM=120000
EVENT_COUNT_PER_JOB=10000
JOB_COUNT_PER_SPILL=30
FIRST=$((((${PROCESS_START} + ${PROCESS}))* ${EVENT_COUNT_PER_JOB}))
LAST=$((${FIRST} + $EVENT_COUNT_PER_JOB - 1))
echo "PROCESS is: $PROCESS"
echo "EVENT_COUNT_PER_JOB is: $EVENT_COUNT_PER_JOB"
echo "FIRST = $FIRST"
echo "LAST = $LAST"
JOB_COUNT=$((${PROCESS_START} + ${PROCESS} + 1))
EVENT_COUNT_PER_SPILL=$((${EVENT_COUNT_PER_JOB} * ${JOB_COUNT_PER_SPILL}))

g4bl beam.py.in first=${FIRST} last=${LAST} particle=${PARTICLE} momentum=${MOMENTUM}
python merge_tree.py beam.root --subspillnumber $JOB_COUNT --subspillcount $JOB_COUNT_PER_SPILL --spillsize $EVENT_COUNT_PER_SPILL

chmod 766 MergedAtstart_linebeam.root
chmod 766 MergedAtstart_linebeam.pickle
ifdh cp MergedAtstart_linebeam.root /pnfs/nova/scratch/users/junting/g4bl.${B_FIELD}/MergedAtstart_linebeam.${JOB_COUNT}.root
ifdh cp MergedAtstart_linebeam.pickle /pnfs/nova/scratch/users/junting/g4bl.${B_FIELD}/MergedAtstart_linebeam.${JOB_COUNT}.pickle
