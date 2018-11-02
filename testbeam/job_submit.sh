jobsub_submit -G nova -N 2000 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --role=Analysis --expected-lifetime=8h file://$PWD/job_script.sh
