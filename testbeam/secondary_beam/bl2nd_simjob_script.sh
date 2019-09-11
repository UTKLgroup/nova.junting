
PROCESS_START=0
EVENT_COUNT_PER_JOB=10000
FIRST=$((((${PROCESS_START} + ${PROCESS}))* ${EVENT_COUNT_PER_JOB}))
LAST=$((${FIRST} + $EVENT_COUNT_PER_JOB - 1))
echo "PROCESS is: $PROCESS"
echo "EVENT_COUNT_PER_JOB is: $EVENT_COUNT_PER_JOB"
echo "FIRST = $FIRST"
echo "LAST = $LAST"

function setup_nova_grid {
  source /cvmfs/nova.opensciencegrid.org/novasoft/slf6/novasoft/setup/setup_nova.sh "$@"
}

setup_nova_grid -r S18-01-19 -6 /cvmfs/nova.opensciencegrid.org/novasoft/slf6/novasoft -e /cvmfs/nova.opensciencegrid.org/externals -b maxopt
setup G4beamline v3_04

export G4BL_DIR=/cvmfs/nova.opensciencegrid.org/externals/G4beamline/v3_04
export G4ABLADATA=/cvmfs/nova.opensciencegrid.org/externals/g4abla/v3_0/NULL/G4ABLA3.0
export G4LEDATA=/cvmfs/nova.opensciencegrid.org/externals/g4emlow/v6_50/G4EMLOW6.50
export G4ENSDFSTATEDATA=/cvmfs/nova.opensciencegrid.org/externals/g4nuclide/v2_1/G4ENSDFSTATE2.1
export G4NEUTRONHPDATA=/cvmfs/nova.opensciencegrid.org/externals/g4neutron/v4_5/NULL/G4NDL4.5
export G4NEUTRONXSDATA=/cvmfs/nova.opensciencegrid.org/externals/g4neutronxs/v1_4/NULL/G4NEUTRONXS1.4
export G4PIIDATA=/cvmfs/nova.opensciencegrid.org/externals/g4pii/v1_3/NULL/G4PII1.3
export G4SAIDXSDATA=/cvmfs/nova.opensciencegrid.org/externals/g4nucleonxs/v1_1/NULL/G4SAIDDATA1.1
export G4LEVELGAMMADATA=/cvmfs/nova.opensciencegrid.org/externals/g4photon/v4_3_2/PhotonEvaporation4.3.2
export G4RADIOACTIVEDATA=/cvmfs/nova.opensciencegrid.org/externals/g4radiative/v5_1_1/RadioactiveDecay5.1.1
export G4REALSURFACEDATA=/cvmfs/nova.opensciencegrid.org/externals/g4surface/v1_0/NULL/RealSurface1.0

ifdh cp /pnfs/nova/persistent/users/dphan/novatestbeam/bl2nd_simulation/input/secondary.watts.dung.allParticle.QGSP_BIC.64GeV.in ./secondary.in

g4bl secondary.in first=${FIRST} last=${LAST}

chmod 766 g4beamline.root
ifdh cp g4beamline.root /pnfs/nova/persistent/users/dphan/novatestbeam/bl2nd_simulation/output/small_production/g4beamline.${FIRST}.root
