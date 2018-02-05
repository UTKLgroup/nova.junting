FIRST=0
LAST=10

MOMENTUM=32
g4bl target.in first=$FIRST last=$LAST momentum=$MOMENTUM
mv -f target.root data/target.${MOMENTUM}GeV.first_${FIRST}.last_${LAST}.root

MOMENTUM=64
g4bl target.in first=$FIRST last=$LAST momentum=$MOMENTUM
mv -f target.root data/target.${MOMENTUM}GeV.first_${FIRST}.last_${LAST}.root
