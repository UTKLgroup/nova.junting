jobsub_submit -G nova -N 2000 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --role=Analysis --expected-lifetime=6h file://$PWD/job_script.sh
