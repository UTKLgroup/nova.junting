MOMENTUM=8

g4bl target.in last=10000 momentum=$MOMENTUM
mv target.root data/target.${MOMENTUM}GeV.tubs.track.fix_angle.root
