jobsub_submit -G nova -N 900 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --role=Analysis --expected-lifetime=2h file://$PWD/job_script.sh
