jobsub_submit -G nova -N 900 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --role=Analysis --expected-lifetime=3h --memory=4GB file://$PWD/job_script.sh
