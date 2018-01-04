SHELL = /bin/sh
NAME = all
MAKEFILE = Makefile

include $(GENIE)/src/make/Make.include

GENIE_LIBS  = $(shell $(GENIE)/src/scripts/setup/genie-config --libs)
LIBRARIES  := $(GENIE_LIBS) $(LIBRARIES) $(CERN_LIBRARIES)

TGT =	gtestEventLoop

all: $(TGT)

gtestEventLoop: FORCE
	$(CXX) $(CXXFLAGS) -c gtestEventLoop.cxx $(INCLUDES)
	$(LD) $(LDFLAGS) gtestEventLoop.o $(LIBRARIES) -o ./gtestEventLoop

purge: FORCE
	$(RM) *.o *~ core 

clean: FORCE
	$(RM) *.o *~ core 
	$(RM) ./gtestEventLoop

FORCE:
# DO NOT DELETE
