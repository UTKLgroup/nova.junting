# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ObjSuf        = o
SrcSuf        = cxx
DllSuf        = so
OutPutOpt     = -o 

SJY_DIR = ${shell pwd}/fnal_camac
INCDIR = ${shell pwd}/include
INC = -I$(SJY_DIR)/include -I$(INCDIR)

LIBDIR = ${shell pwd}/lib

ROOTCFLAGS   := -g $(INC) $D -DNEW -Wall -O2 $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --new --libs) 
ROOTGLIBS    := $(shell root-config --glibs)

CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O -L$(SJY_DIR)/lib -L$(LIBDIR) 

CXXFLAGS     += $(ROOTCFLAGS)
LIBS          = $(ROOTLIBS) -lsjy -lConfig -lkvp -lMsgLog

#------------------------------------------------------------------------------

CAMACDAQO      = camacDaq.$(ObjSuf)
CAMACDAQS      = camacDaq.$(SrcSuf)
CAMACDAQ       = camacDaq
OBJS          = $(CAMACDAQO) 
PROGRAMS      = $(CAMACDAQ) 

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) 

all:            $(PROGRAMS)

$(CAMACDAQ):     $(CAMACDAQO)
		$(LD)  $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

clean:
		@rm -f $(OBJS) camacDaq 


.SUFFIXES: .$(SrcSuf)

###

.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
