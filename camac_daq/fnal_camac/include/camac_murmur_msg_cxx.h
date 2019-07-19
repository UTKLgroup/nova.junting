#ifndef CAMAC_MURMUR_MSG_CXX_HEADER
#define CAMAC_MURMUR_MSG_CXX_HEADER
//*****************************************************************
//**
//**                  MURMUR Message Utility
//**
//** File: ../include/camac_murmur_msg_cxx.h
//** Created: Mon Oct 16 10:49:12 1995
//** Message Input File: ../include/camac_murmur.msg
//** By User: streets
//**
//******************************************************************
//  CAMAC Murmur message definitions	das 1aug95
// 
//  29aug95 das	added VME_CLOSE_FAIL for cdchn( ,FALSE, )
//  11oct95 jms	made NO_Q informational
// 		added SCSI_CLOSE_FAIL
const unsigned int	CAM_facnum = 0X95;
const	unsigned int	CAM_S_SUCCESS = 0X80254009;
const	unsigned int	CAM_S_NO_Q = 0X80254013;
const	unsigned int	CAM_S_INVLD_CRATE = 0X8025401C;
const	unsigned int	CAM_S_INVLD_ARG = 0X80254024;
const	unsigned int	CAM_S_NO_DMD = 0X8025402C;
const	unsigned int	CAM_S_INVLD_DMD = 0X80254034;
const	unsigned int	CAM_S_NO_RESPONSE = 0X8025403C;
const	unsigned int	CAM_S_NO_X = 0X80254044;
const	unsigned int	CAM_S_VME_OPEN_FAIL = 0X8025404C;
const	unsigned int	CAM_S_MMAP_FAIL = 0X80254054;
const	unsigned int	CAM_S_DEV_OPEN_FAIL = 0X8025405C;
const	unsigned int	CAM_S_DEV_IO_ERROR = 0X80254064;
const	unsigned int	CAM_S_SEM_FAIL = 0X8025406C;
const	unsigned int	CAM_S_GETSEM_FAIL = 0X80254074;
const	unsigned int	CAM_S_PUTSEM_FAIL = 0X8025407C;
const	unsigned int	CAM_S_INVLD_BRANCH = 0X80254084;
const	unsigned int	CAM_S_PARALLEL = 0X8025408C;
const	unsigned int	CAM_S_SER_OR_PAR_ONLY = 0X80254094;
const	unsigned int	CAM_S_RPC_FAIL = 0X8025409C;
const	unsigned int	CAM_S_NO_ROUTE = 0X802540A4;
const	unsigned int	CAM_S_INVLD_SLOT = 0X802540AC;
const	unsigned int	CAM_S_INVLD_SUBADD = 0X802540B4;
const	unsigned int	CAM_S_VME_CLOSE_FAIL = 0X802540BC;
const	unsigned int	CAM_S_SCSI_CLOSE_FAIL = 0X802540C4;
#endif
