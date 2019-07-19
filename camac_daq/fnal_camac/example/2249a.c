/* 
** 2249a - read out the 12 pedestals of a LeCroy LRS2249A ADC after 
**         generating a test pulse with the test module function.
**         Uses Q-Stop or single word readout mode.
**       
** 11may99 das
** 23may02 das removed unused variables sec, status; added stdlib.h include
**             for exit
**
**/

#include "ieee_fun_types.h"
#include <stdio.h>
#include <stdlib.h>
#include "timekit.h"

#define NWORDS 12

int main(int argc, char **argv) {

    int	   ext;
    int	   branch;
    int    crate;
    int    slot;
    int	   type;
    int    subad=0;
    unsigned short dataw[2]={0,0};
    unsigned short data[NWORDS];

    int	   block[4]={0};
    int	   i,j,m;
    int	   qres=0;




    if(argc < 5) {
      printf("Usage: 2249a <branch> <crate> <slot> <type: serial=0,parallel=1>\n");
      return 0;
    }  


    if(argc > 1)
	sscanf(argv[1],"%d",&branch);

    if(argc > 2)                           
        sscanf(argv[2],"%d",&crate);

    if(argc > 3)
	sscanf(argv[3],"%d",&slot);

    if(argc > 4)                           
        sscanf(argv[4],"%d",&type);

    /* open the SCSI device */
    if((cdchn(branch,1,0) &1) != 1){
	perror("cdchn error");
	exit(2);
	}

    /* set the controller type */
    if(type == 0)
	ccctype(branch,1,0); /* serial */
    else
        ccctype(branch,0,1); /* parallel - Jor 73A is considered parallel */


    /* encode the branch, crate, slot, 
       and subad arguments into ext */
    cdreg(&ext,branch,crate,slot,subad);

    /* crate clear */
    cccc(ext);
    /* crate initialize */
    cccz(ext);
    /* remove crate inhibit */
    ccci(ext,0);


    /* clear 2249A LAM */
    cssa(26,ext,dataw,&qres);


    block[0]=NWORDS;   /* set the transfer size */


    /* initiate 2249A test pulse */
    cssa(25,ext,dataw,&qres);

    /* poll for LAM */
    cssa(8,ext,dataw,&qres);
    while(!qres) {
      for (i=0; i<5000; i++){};
      cssa(8,ext,dataw,&qres);
    }



#if 0
    /* do a test readout since the F2 read to 
       chan 11 clears the 2249A */
    csmad(2,ext,data,&block[0]); 
    printf("0:%2d  1:%2d  2:%2d  3:%2d   4:%2d   5:%2d\n6:%2d  7:%2d  8:%2d  9:%2d  10:%2d  11:%2d\n",
         data[0],data[1],data[2],data[3],data[4],data[5], 
 	   data[6],data[7],data[8],data[9],
         data[10],data[11]); 
#endif


    /* timing loop */
    m = 100000;
    START_TIME;
    for (j=0; j<m; j++) {


#ifdef SINGLEWORDREADOUT
      for (subad=0; subad<11; subad++) {
	  cdreg(&ext,branch,crate,slot,subad);
	  cssa(2,ext,&data[subad],&qres);
	  if (!qres) 
	    printf ("no Q from slot %d sub-address %d\n",slot,subad);
      }
#else
      /* auto increment sub-address and slot when Q=0 */
      csmad(2,ext,data,&block[0]);
#endif

    } /* end timing loop */
    END_TIME;
    TOT_TIME;



    printf("Time elapsed for %d repetitions of length %d words\n",
 	   m, NWORDS);
/*     PRINT_TIME(stdout); */
    printf("Total time = %0.6f  Time per word = %0.6f\n",ftime, ftime/(12*m));

    return 0;
}













