/**
 ** cz.c - execute a CAMAC crate initialization command
 **
 ** 24may2002 das removed unused variables n,function; added stdlib.h include
 **               for exit
 */


#include "ieee_fun_types.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {

    int	ext;
    int branch;
    int	crate;
    int	slot=10;
    int	subad=0;
    int type;


    if(argc < 4) {
      printf("Usage: cz <branch> <crate> <type: serial=0,parallel=1>\n");
      return 0;
    }  

    if(argc > 1)                           
        sscanf(argv[1],"%d",&branch);

    if(argc > 2)                           
        sscanf(argv[2],"%d",&crate);

    if(argc > 3)
	sscanf(argv[3],"%d",&type);

    if((cdchn(branch,1,1) &1) != 1){
	perror("cdchn error");
	exit(1);
	}

    if(type==1)
      ccctype(branch,0,1);
    else
      ccctype(branch,1,0);

    cdreg(&ext,branch,crate,slot,subad);
    cccbyp(ext,0);
    cccoff(ext,0);
    ccci(ext,0);
    cccc(ext);
    cccz(ext);
    ccci(ext,0);
    cmsg("cz: ",0,branch);
    exit(0);
}



