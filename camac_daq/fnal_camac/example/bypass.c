/**
 ** bypass.c - Utility to bypass or unbypass a serial crate controller.  
 **            Not used for the Jor 73A, but retained for compatiblity 
 **            with other CAMAC packages.
 ** 
 ** 24may2002 das removed unused variable qres, added stdlib.h include
 **               for exit
 */

#include "ieee_fun_types.h"
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv) {

    int	ext;
    int	branch;
    int	crate;
    int	slot=1;
    int	subad=0;
    int onoff;
    int status;

    if(argc < 4) {
      printf("Usage: bypass <branch> <crate> <on=1, off=0>\n");
      return 0;
    }  

    if(argc > 1)                           
        sscanf(argv[1],"%d",&branch);

    if(argc > 2)                           
        sscanf(argv[2],"%d",&crate);

    if(argc > 3)
	sscanf(argv[3],"%d",&onoff);


    if((cdchn(branch,1,0) &1) != 1){
	perror("cdchn:");
	exit(2);
	}

    cdreg(&ext,branch,crate,slot,subad);
    cccbyp(ext,onoff);

    status=cmsg("Bypass",0,branch);

    exit(0);
}




