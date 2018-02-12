
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

ifdh cp /pnfs/nova/persistent/users/junting/testbeam/merge_tree.py ./merge_tree.py
ifdh cp /pnfs/nova/persistent/users/junting/testbeam/beam.in ./beam.in
g4bl beam.in
python merge_tree.py beam.root

chmod 766 beam.root
chmod 766 MergedAtStartLinebeam.root
ifdh cp beam.root /pnfs/nova/scratch/users/junting/g4bl/beam.root
ifdh cp MergedAtStartLinebeam.root /pnfs/nova/scratch/users/junting/g4bl/MergedAtStartLinebeam.root
