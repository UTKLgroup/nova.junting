/**
 ** cnaf.c - execute a single CAMAC command
 **
 ** arguments:
 ** branch - scsi_Id + 8*(scsi_bus). The thumb wheel on the Jor 73A front
 **          panel, "Crate Address" sets its SCSI Id. (Note though that you
 **          can't change this setting once your computer has booted. It
 **          would be like changing the SCSI Id of an active SCSI disk drive:
 **          the system will no longer recognize it until the next reboot.) 
 **	     The branch calculation is  the Jor73A "Crate Address" 
 **	     (which must be in the range 0-6) + 8*(scsi_bus). 
 **	     "Crate Address" 7 is never used because the AHA-2490AU and
 **          other SCSI host controller cards use SCSI Id 7. You can confirm 
 **          what your SCSI bus and Jor 73A SCSI bus Id are by doing a more 
 **          on  /proc/scsi/scsi
 **
 ** crate -  has no meaning for the Jor 73A but is kept for compatibility 
 **          with other CAMAC packages. The crate number must still be in the 
 **          valid range of crate numbers, so use "1" here. 0 is not valid.
 **
 ** slot -   the crate slot position that is the target of the cnaf command
 **
 ** subad -  the subaddress to use in the cnaf command
 ** 
 ** function - the function number to use in the cnaf command
 **
 ** data -   the data if the cnaf command is a write function. The data 
 **          argument is given as 0 for non-data or read functions
 **
 ** type -   always 1 for parallel. The Jor73A emulates the parallel controller
 **          command set
 **
 **
 ** 24may2002 das removed unused variables cb,route,n,status,i; added
 **               include stdlib.h for exit
 **/


#include "ieee_fun_types.h"
#include <stdio.h>
#include <stdlib.h>

#define BIT16 1
int main(int argc, char **argv) {

    int	ext;
    int branch;
    int	crate;
    int	slot;
    int	subad;
    int	function;
    int temp;
#ifdef BIT16
    unsigned short data[100] = {0};
#else
    int data[100] = {0};
#endif
    int	qres;
    int type;


    if(argc < 8) {
      printf("Usage: cnaf <branch> <crate> <slot> <subad> <function> <data> <type: serial=0,parallel=1>\n");
      return 0;
    }  


    if(argc > 1)
	sscanf(argv[1],"%d",&branch);

    if(argc > 2)                           
        sscanf(argv[2],"%d",&crate);

    if(argc > 3)
	sscanf(argv[3],"%d",&slot);

    if(argc > 4)
	sscanf(argv[4],"%d",&subad);

    if(argc > 5)
	sscanf(argv[5],"%d",&function);

    if(argc > 6)
	sscanf(argv[6],"%d",&temp);

    if(argc > 7)                           
        sscanf(argv[7],"%d",&type);


    data[0] = temp;

    if((cdchn(branch,1,0) &1) != 1){
	perror("cdchn error");
	exit(1);
	}

    if(type == 0)
	ccctype(branch,1,0); /* serial */
    else
        ccctype(branch,0,1); /* parallel - Jor 73A is considered parallel */

    cdreg(&ext,branch,crate,slot,subad);
/*
    cccbyp(ext,0);
    cccoff(ext,0);
    ccci(ext,0);
    cccc(ext);
    cccz(ext);
    ccci(ext,0);
*/
/*  for(i=0;i<10;i++)*/
#ifdef BIT16
        cssa(function,ext,&data[0],&qres);
/*      csubc(function,ext,&data[0],cb);*/
#else 
        cfsa(function,ext,&data[0],&qres);
/*      cfubc(function,ext,&data[0],cb);*/
#endif

    cmsg("cnaf:",data[0],branch);
/*
    if(!qres){
      printf("Data=0x%x, Q=%d, NW=%d\n",*data,qres,cb[1]);
      cmsg("cnaf: Error",0,branch);
     }
*/

    exit(0);
}


