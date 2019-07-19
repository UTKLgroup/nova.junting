/**
 ** pdt.c - loops continuously writing an incrementing integer, reading
 **         it back, and checking for data integrity
 **
 ** 23May2002 das removed unused variable status, added stdlib.h include
 **               for exit
 */

#include "ieee_fun_types.h"
#include <stdio.h>
#include <stdlib.h>

/*#define BIT16*/           /* 16 bit data words */
/*#define MULTIUTEST*/      /* mutli-user test   */

extern SJY_CONTROLLER sjy_controller[];

int main(int argc, char **argv) {

    int	ext;
    int	branch;
    int	crate;
    int	slot;
    int	subad=0;
#ifdef BIT16
    unsigned short	data1[2];
    unsigned short	data2[2];
#define DATA_MASK	0xFFFF
#else
    int data1[1];
    int data2[1];
#define DATA_MASK	0xFFFFFF
#endif
    int	qres1;
    int	qres2;
    int	n=0;
    int	type;

    if(argc < 5) {
      printf("Usage: pdt <branch> <crate> <slot> <type: serial=0,parallel=1>\n");
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


    if((cdchn(branch,1,1) &1) != 1){
	perror("cdchn error");
	exit(2);
	}

    if(type == 0)
	ccctype(branch,1,0); /* serial */
    else
        ccctype(branch,0,1); /* parallel - Jor 73A is considered parallel */

    cdreg(&ext,branch,crate,slot,subad);

#ifdef DEBUG
    printf("%s\n", sjy_inquiry(sjy_controller[IDX(branch)].fd) + 8);
#endif

#ifndef MULTIUTEST
    cccbyp(ext,0);
    cccoff(ext,0);
    ccci(ext,0);
    cccc(ext);
    cccz(ext);
    ccci(ext,0);
#endif
    *data1=1;

    *data2=*data1;
    qres1=1;
    qres2=1;

    while (*data1 == *data2 ) {
	*data1+=1;
	*data1 = (*data1&DATA_MASK);
#ifdef BIT16
	cssa(16,ext,data1,&qres1);
	cssa(0,ext,data2,&qres2);
#else
	cfsa(16,ext,data1,&qres1);
	cfsa(0,ext,data2,&qres2);
#endif
	n++;
	if(n%10000 == 0){
	  printf(" %d pdt's complete \r",n);
	  fflush(stdout);
	  }
    }
    printf("\npdt: Data mismatch on %d, wrote 0x%x(Q=%d), read 0x%x(Q=%d)\n",
	n,*data1,qres1,*data2,qres2);

    cmsg("pdt: Word Mismatch",0,branch);
    return 0;
}













