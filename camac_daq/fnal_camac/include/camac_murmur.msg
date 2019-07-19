# CAMAC Murmur message definitions	das 1aug95
#
# 29aug95 das	added VME_CLOSE_FAIL for cdchn( ,FALSE, )
# 11oct95 jms	made NO_Q informational
#		added SCSI_CLOSE_FAIL


-facility CAM 149 -prefix CAM_S

SUCCESS 	-severity S 	<Successful Execution>
NO_Q		-severity I 	<no Q from CAMAC slot>
INVLD_CRATE	-severity E	<Serial must be between 1 and 62; Parallel between 1 and 7>
INVLD_ARG	-severity E	<IEEE routine doesnt like this argument>
NO_DMD		-severity E	<no LAMs present>
INVLD_DMD	-severity E	<not a TRUE demand message>
NO_RESPONSE	-severity E	<no response from crate controller>
NO_X		-severity E	<no X from CAMAC slot>
VME_OPEN_FAIL	-severity E	<Couldnt open VME device file>
MMAP_FAIL	-severity E	<Couldnt map to VME device fail>
DEV_OPEN_FAIL	-severity E	<Could not open JY411S file in /dev/sjycNsMl0>
DEV_IO_ERROR	-severity E	<device error from read or write>
SEM_FAIL	-severity E	<Couldnt initialise Semaphore>
GETSEM_FAIL	-severity E	<Couldnt Take Semaphore>
PUTSEM_FAIL	-severity E	<Couldnt Release Semaphore>
INVLD_BRANCH	-severity E	<Branch number must be between 0 and 255 (VSD2992), 0 and 7 (CES8210), 0 and SJY_MAX_JY411 (JY411S)>
PARALLEL	-severity E	<Branch is Parallel>
SER_OR_PAR_ONLY	-severity E	<JY411S must either be parallel or serial, open it again>
RPC_FAIL	-severity E	<Remote Procedure Call fail>
NO_ROUTE	-severity E	<Route must be initialised with cdroute()>
INVLD_SLOT	-severity E	<Slot must be between 1 and 30>
INVLD_SUBADD	-severity E	<Subaddress must be between 0 and 15>
VME_CLOSE_FAIL	-severity E	<Couldnt close VME device file>
SCSI_CLOSE_FAIL	-severity E	<Couldnt close SCSI device file>


-end
