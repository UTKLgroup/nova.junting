jobsub_submit -G nova -N 500 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --role=Analysis --expected-lifetime=5h file:///nova/app/users/dphan/nova.junting/testbeam/secondary_beam/bl2nd_simjob_script.sh
