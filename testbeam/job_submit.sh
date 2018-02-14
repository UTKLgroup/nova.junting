jobsub_submit -G nova -N 1 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC --role=Analysis --expected-lifetime=120m --memory=1000MB file://$PWD/job_script.sh
