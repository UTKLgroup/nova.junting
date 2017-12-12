source /nova/app/users/junting/g4beamline/G4beamline-3.04/bin/g4bl-setup.sh
source /grid/fermiapp/products/common/etc/setups.sh
setup jobsub_client
setup ifdhc

jobsize=10000
first=$((${PROCESS} * ${jobsize}))
last=$((${first} + $jobsize - 1))

ifdh cp /nova/app/users/junting/testbeam/beam/beam.in input
g4bl input first=$first last=$last

chmod 777 beam.root
ifdh cp beam.root /pnfs/nova/scratch/users/junting/g4bl/beam_${PROCESS}.root
