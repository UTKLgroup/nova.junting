/*  ============================================================================
**
**   Computing Department
**   Fermilab, Batavia, Il 60510, U.S.A.
**
**   Description
**   -----------
**   C language tool kit for cross platform timing tests.
**	platform	resolution
**	VAX		sec
**	SUN		usec
**	SGI		usec
**	MVME167		usec
**   The user interface for all platforms is:
**	START_TIME		Records system time
**	END_TIME		Records system time
**	PRINT_TIME(FILE *file)	Prints delta time, FILE * will probably be
**				stdio.h "stdout" in most cases
**
**   Side effects
**   ------------
**   Global declarations for macro arguments.
**
**   Written / Modified by :
**   -----------------------
**   23-Sep-1992	DAS	Original
**   27-Oct-1992	DAS	Changing macros to include AIX and ULTRIX
**
**   ===========================================================================
**
*/
/* header files */
#ifdef VMS

#include <stdio.h>
#include <stdlib.h>

#elif VxWORKS

#include <stdioLib.h>

#include "mvsup.h"


#else

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#endif

/* declarations/initializations */
	FILE	*stream;

#ifdef VMS
	int	time_a,time_b;
	float	dtime;
	int	clock_a,clock_b;
	float	dclock;
#elif VxWORKS
	unsigned int t1, t2;
#else
	struct itimerval time1, time2;
	long int  dtime_sec, time_a_sec, time_b_sec;
	long int  dtime_usec, time_a_usec, time_b_usec;
        float ftime;  
#endif


/* macros */
#ifdef VMS
#define S_TIME(time_a, clock_a) {time_a = time(); clock_a = clock();}
#define START_TIME 	S_TIME(time_a, clock_a)
#define E_TIME(time_b, clock_b) {time_b = time(); clock_b = clock();}
#define END_TIME	E_TIME(time_b, clock_b)
#define DIF_TIME(dtime, time_b, time_a) {dtime = time_b - time_a;}
#define DIF_CLOCK(dclock, clock_b, clock_a) {dclock = clock_b - clock_a;}
#define PRN_TIME(stream, dtime, dclock) \
         {DIF_TIME(dtime, time_b, time_a) \
	  DIF_CLOCK(dclock, clock_b, clock_a) \
	  if (dtime > 0) \
             fprintf(stream, \
             " %.0f secs (%.0f ms CPU)\n", \
             dtime, dclock*10.); \
          else \
             puts("\t difference time not greater than 0");}
#define PRINT_TIME(stream)	PRN_TIME(stream, dtime, dclock)

#elif VxWORKS
#define S_TIME(t1, t2) \
	{t1 = 0; t2 = 0; \
	 clk2Stop(); \
	 clk2Start(); \
	 t1 = clk2Read;}
#define START_TIME	S_TIME(t1, t2)
#define E_TIME(t1, t2) \
	{t2 = clk2Diff(t1);}
#define END_TIME	E_TIME(t1, t2)
#define PRN_TIME(stream, t2) \
	{fprintf(stream, " %d usec\n", t2);} 
#define PRINT_TIME(stream)	PRN_TIME(stream, t2)

#else
#define S_TIME(time1, time2) \
	{time1.it_value.tv_sec = 100000; 	/* arbitrarily large */ \
	 time1.it_value.tv_usec = 0; \
	 time1.it_interval.tv_sec = 1000;	/* also arbitrary */ \
	 time1.it_interval.tv_usec = 0; \
	 setitimer(0, &time1, &time2); \
	 getitimer(0, &time1); \
	 time_a_sec  =  time1.it_value.tv_sec; \
	 time_a_usec =  time1.it_value.tv_usec;}
#define START_TIME	S_TIME(time1, time2)
#define E_TIME(time1, time_b_sec, time_b_usec) \
	{getitimer(0, &time1); \
	 time_b_sec = time1.it_value.tv_sec; \
	 time_b_usec = time1.it_value.tv_usec;}
#define END_TIME	E_TIME(time1, time_b_sec, time_b_usec)
#define DIF_TIME(dtime_sec, time_b_sec, time_a_sec, dtime_usec, time_b_usec, time_a_usec) \
	{dtime_sec  = abs(time_b_sec - time_a_sec); \
	 dtime_usec = abs(time_b_usec - time_a_usec);}
#define PRN_TIME(stream, dtime_sec, dtime_usec) \
         {DIF_TIME(dtime_sec, time_b_sec, time_a_sec, dtime_usec, time_b_usec, time_a_usec) \
          fprintf(stream, " %d secs  %d usec\n", \
	  dtime_sec, dtime_usec);}
#define PRINT_TIME(stream)	PRN_TIME(stream, dtime_sec, dtime_usec)
#define TOT_TIME \
          {DIF_TIME(dtime_sec, time_b_sec, time_a_sec, dtime_usec, time_b_usec, time_a_usec); \
          ftime = dtime_sec; \
          ftime = ftime + (dtime_usec/1000000.0);}
#endif
