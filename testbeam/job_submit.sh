# jobsub_submit.py -G nova --memory=4GB --expected-lifetime=25h -N 2 --OS=SL5,SL6 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC file://$PWD/Script.sh
jobsub_submit.py -G nova -N 360 --OS=SL5,SL6 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC file://$PWD/Script.sh
