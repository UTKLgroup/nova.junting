/* sjy_semops.c				jms 6sep94 */
/* 21aug97 das Modified vsd_semops for Linux CAMAC */
/* 25sep98 das Stubbed for performance improvement test */

/* operations on the semaphores - init, get and put. 
 * sjy_inisem()		called in inicam
 * sjy_getsem,putsem	called to reserve the semaphore
 *
 */


int sjy_inisem()
{
   return(1);
}

int sjy_getsem()
{
/* get the sjy semaphore */
  return(0);
}

int sjy_putsem()
{
/* release the sjy semaphore */
  return(0);
}



















