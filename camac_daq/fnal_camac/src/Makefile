# CFLAGS must define either SJY411S or SJY73A

#CC = g++

CFLAGS = -g -Wall -I../include -DSJY73A

LIBRARY = ../lib/libsjy.a

all	: $(LIBRARY)

INCLUDES = ../include/camac_murmur_msg_c.h \
	../include/ieee_fun_types.h ../include/camac_globals.h ../include/sjy.h

../include/ieee_fun_types.h : $(CAMAC_DIR)/include/ieee_fun_types.h
	cp $(CAMAC_DIR)/include/ieee_fun_types.h $@

../include/camac_murmur_msg_c.h : ../include/camac_murmur.msg
	$(MURMUR_DIR)/bin/murmessage ../include/camac_murmur.msg ../lib

../include/camac_murmur.msg : $(CAMAC_DIR)/include/camac_murmur.msg
	cp $(CAMAC_DIR)/include/camac_murmur.msg $@

OBJS =  sjy_cdchn.o sjy_cccd.o sjy_cccc.o sjy_ccci.o sjy_cccz.o sjy_cssa.o \
sjy_cfsa.o sjy_cdreg.o sjy_cmsg.o sjy_ctgl.o sjy_cclm.o sjy_cgreg.o \
sjy_cccbyp.o sjy_cccoff.o sjy_ctstat.o sjy_csmad.o sjy_cfmad.o sjy_csubc.o \
sjy_cfubc.o sjy_camerr.o sjy_cderr.o sjy_ccctype.o sjy_csuns.o \
sjy_xrespn.o sjy_qrespn.o sjy_get_qx.o sjy_cdroute.o sjy_reroute.o\
sjy_interface.o sjy_semops.o

#$(OBJS)	: $(INCLUDES)

$(LIBRARY)	: $(OBJS) 
	rm -f $(LIBRARY);	\
	$(AR) -r $(LIBRARY) $(OBJS)
	rm -f $(OBJS)



