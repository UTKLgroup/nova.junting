/******************************************************************************
qxtab.c - print table of X and Q responses for a CAMAC module

By: agatam
Data: 21-sep-94
      23-may-02 das added stdlib.h include for exit
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ieee_fun_types.h"

  char line[100], stResp[][4] = { " S", " X", " Q", "  " };
  int k, resp;
  int return_status = CAM_S_SUCCESS;
  int branch = 5;
  int crate = 5;
  int slot = 4;
  int type = 1; 
  int ext, subad, fun, data;

int main(int argc, char **argv)
{

  int status;

  if(argc < 5) {
    printf("Usage: qxtab <branch> <crate> <slot> <type: serial=0,parallel=1>\n");
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
  /*
  printf("\n");
  printf(" Select branch(in dec)  [%d] : ",branch);  scanf("%d", &branch);
  printf(" Select crate (in dec)  [%d] : ",crate);  scanf("%d", &crate);
  printf(" Select slot  (in dec)  [%d] : ",slot );  scanf("%d", &slot);
  printf(" Serial (0) or Parallel(1) [%d] : ",type); scanf("%d", &type);
  */
  if ((return_status = cdchn(branch, 1, 0)) !=  CAM_S_SUCCESS)  {
    printf("Error opening branch=%d\n",branch);
    exit(1);
  };

  cdreg(&ext, branch,crate,slot,subad);       
  if(type == 0)
	ccctype(branch,1,0);  /* serial */
  else
	ccctype(branch,0,1);  /* parallel - Jor 73A is considered parallel */

  cccbyp(ext,0);                       
  cccoff(ext,0);                       
  ccci(ext,0);                         
  cccc(ext);                           
  cccz(ext);                           
  ccci(ext,0);                         

  printf("\n\n  S == Q=1 and X=1  ");
  printf
   ("\nF                         1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3");
  printf
   ("\n      0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1");
  printf("\nA\n\n");
  for (subad = 0; subad <= 15; subad++) {
    cdreg(&ext, branch, crate, slot, subad);
    strcpy(line, "  ");  
    for (fun = 0; fun <= 31; fun++) {
      status=cfsa(fun, ext, &data, &resp);
      status=ctstat( &k );
      strcat(line, stResp[k & 3]);
    };
    printf("\n%2d %s", subad, line);
  };
  printf("\n");
  return 0;
}    

