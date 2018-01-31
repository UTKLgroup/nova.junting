source /grid/fermiapp/products/common/etc/setups.sh
setup jobsub_client
setup ifdhc

source /grid/fermiapp/nova/novaart/novasvn/setup/setup_nova.sh
cd /nova/app/users/junting/nnbar
srt_setup -a

nova -n 3 -c prodgenie_nosc_cosmic.fcl
ifdh cp nnbar_merge.root /pnfs/nova/scratch/users/junting/nnbar/test_${PROCESS}.root
