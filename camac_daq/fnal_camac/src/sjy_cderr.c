/*  
** 23-May-2002 das Added 0 return value 
*/


#include "ieee_fun_types.h"

unsigned int cderr (int lexit,
	            int	llog,
	            int	ltrace)
{
    /* This is a dummy subroutine */
  return (0);
}

void cderr_ (int       *lexit,
             int       *llog,
             int       *ltrace)
{
    cderr(*lexit,*llog,*ltrace);
}
