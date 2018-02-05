EVENT_COUNT=150000

MOMENTUM=32
g4bl target.in last=$EVENT_COUNT momentum=$MOMENTUM
mv -f target.root data/target.${MOMENTUM}GeV.final.root

MOMENTUM=64
g4bl target.in last=$EVENT_COUNT momentum=$MOMENTUM
mv -f target.root data/target.${MOMENTUM}GeV.final.root
