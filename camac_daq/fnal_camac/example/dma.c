/* 
** dma.c - Test 16 or 24 bit block read operations using the Fermi Bench
**         Mark module.
**         The Fermi Bench Mark module returns incrementing integers to
**         a block read F2 command, so the data checking portion of this
**         test will show data mismatches with other CAMAC modules.
**
** 19feb98 das Corrected block[2] to block[4], fixed error display indexes
** 23may02 das removed unused variables route,m,status, fixed print formatting
**             for n
**
**/

#include "ieee_fun_types.h"
#include <stdio.h>
#include "timekit.h"

/*#define TIME*/
/*#define BIT16*/           /* use 16 bit words */

                        /* 32768 */    /* 36 */
/* largest buffer size (SG_BIG_BUFF - (SCSI_OFF + 18 + 511)) = 32203 */ 
#define NBYTES 8192        /* 32200, _8192_, 512, 32, 4 */

#ifdef BIT16
#define NWORDS    (NBYTES/2)
#else
#define NWORDS    (NBYTES/4)
#endif



int main(int argc, char **argv) {

    int	   ext;
    int	   branch;
    int    crate;
    int    slot;
    int	   type;
    int    subad=0;
#ifdef BIT16
    unsigned short dataw[2]={0,0};
    unsigned short data[NWORDS];
#define DATA_MASK 0xFFFF
#else
    int	dataw=0;
    int data[NWORDS]={0};
#define DATA_MASK 0xFFFFFF
#endif
    int	        block[4]={0};
    int		err=0;
    unsigned long int n=0;
    int		i,j,k;
    int		qres=0;

    if(argc < 5) {
      printf("Usage: dma <branch> <crate> <slot> <type: serial=0,parallel=1>\n");
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

    if((cdchn(branch,1,0) &1) != 1){
	perror("cdchn");
	exit(1);
	}

    if(type==0) 
      ccctype(branch,1,0);  /* serial */
    else
      ccctype(branch,0,1);  /* parallel -Jor 73A is considered parallel */

    cdreg(&ext,branch,crate,slot,subad);

    cccbyp(ext,0);
    cccoff(ext,0);
    ccci(ext,0);
    cccc(ext);
    cccz(ext);
    ccci(ext,0);


    n=0;
    err=0;

    /* A 0 transfer length  was tried to characterize the dma setup times. 
    ** It seemed to work ok for the Jorway 411S with both serial and parallel 
    ** controllers, but the Jorway 73A would hang in the SCSI read in the 
    ** first csubc call. The Jorway 73A and 411A manuals say "All transfer
    ** modes terminate if the transfer length is satisfied or an error
    ** occurs". Since neither occurs with a 0 length transfer, the 73A hang
    ** is probably alright.
#define NWORDS 0
    */    

    while (err==0){
        block[0]=NWORDS;   /* set the transfer size */
        for(i=0;i<NWORDS;i++) 
		data[i]=0;

#ifdef TIME
	m = 10000;
	err = 1; /* break out of while loop */
	START_TIME;
        for (j=0; j<m; j++) {
#endif

	/* reset data register */
	/* read NWORDS */
#ifdef BIT16
	cssa(16,ext,dataw,&qres);
	/*cssa(9,ext,&data[0],&qres);*/
	csubc(2,ext,&data[0],&block[0]);	/* fix A, N */
#else
	cfsa(16,ext,&dataw,&qres);
        if(!qres){
	  printf("No Q response from setting word\n");
	  err++;
	  break;
	  }
	cfubc(2,ext,&data[0],&block[0]);	/* fix A, N */
#endif

#ifdef TIME
	/* timing stuff */
        } /* end for loop */
	END_TIME;
        printf("Time elapsed for %d repetitions of buffer length of %d words\n",
                m, NWORDS);
	PRINT_TIME(stdout);
        return 0;
#endif

	if(block[1] != NWORDS){
	  printf(" \ndma: loop %ld;read %d, not %d words\n",n,block[1],NWORDS);
	  err++;
	  }
	for(i=0;i<NWORDS;i++){
	  if((data[i]) != (i&DATA_MASK)){
	    err++;
	    printf("loop %ld word 0x%x bad, 10 words around this were:\n",n,i);
	    for(j=i-5;j<i+5;j++) {
              if (j<0)
	        k=j + NWORDS;  /* correction for i=0 */
	      else
		k=j;

		printf("         i= %d data[i]= 0x%x\n",k,data[k]);
            }
	    break;
	    }
	  }
	n++;
	if(n%1000 == 0){
	  printf(" %ld dmas complete \r",n);
	  fflush(stdout);
	  }
     }
     
     cmsg("dma: bad word count",0,branch);
     printf("DMA fail at loop=%ld\n",n);
     fflush(stdout);
     return 0;
}













