/* ************************************************************************* */
/* ===================================================================== 
**
** SUBROUTINE	CDCHN
**
** DESCRIPTION
**	Subroutine to initialize CAMAC environment
**	It opens the scsi device file and establish
**	the mapping between the object file and the user application.
**
** DEVELOPERS
**      Margherita Vittone
**      Fermilab Data Acquisition Group
**      Batavia, Il 60510, U.S.A.
**
**      Eric G. Stern
**      Nevis Labs
**
** MODIFICATIONS
**         Date       Initials  Description
**      21-Aug-1991	MVW     Creation
**	24-Sep-1992	EGS	Modified for MVME147/167 and VxWorks
**	21-Oct-1992	DAS	Modified for VSD2992
**	01-Aug-1994	JAC	IRIX 5, VxWorks
**	06-Sep-1994	JMS	Add semaphore, >1 vsd
**	16-Sep-1994	JMS	JY411S
**	13-Sep-1995	DAS	Close sjy_file if channel=0
**	21-Sep-1995	DAS	Check fd status with ioctl before close 
**      05-Aug-1997     das     modified for Linux
**      31-Mar-2000     das     modified to decode SCSI bus
**      01-Nov-2001     das     added Thomas Hadig bug fix for
**                              multiple controllers
**      23-May-2002     das     removed unused variables version and dev,
**                              added unistd.h,string.h includes
**      11-Apr-2003     das     changed status type from int to ui for g++
**
**  ======================================================================== */
/*									     */

/*#define DEBUG*/

#define XDEFLG
#include "ieee_fun_types.h"		/* ieee_status declared in this file */
#undef XDEFLG

#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

int sjy_getdev (char *dev, int branch);


unsigned int cdchn (int	branch,
	            int	channel,
	            int	route)

{
/* -----------------------------------------------------------------------   */
  char	filename[72];
  int	fd;
  unsigned int status;
  struct stat buf;
  int idx = IDX(branch);


  if( ID(branch)<0 || ID(branch)>SJY_BRANCH_MAX )
	return(CAM_S_INVLD_BRANCH);

  /* close the SCSI device file */
  if (channel == 0) {  
    if (fstat(sjy_controller[idx].fd, &buf) != -1) {
      close(sjy_controller[idx].fd);
      return(CAM_S_SUCCESS);
    }
  return(CAM_S_SCSI_CLOSE_FAIL);
  }

  /* find the scsi device name based on our SCSI id and SCSI bus */
  status = sjy_getdev (filename, branch);
  if (status != CAM_S_SUCCESS) {
    fprintf(stderr, "CDCHN Error finding generic scsi device\n");
    return(status);
  }

  /* create the semaphore */
  if(sjy_inisem() == -1){
    perror("cdchn: inisem");
    return(CAM_S_SEM_FAIL);
  }

  /* reserve the semaphore */
  if(sjy_getsem() != 0){
    perror("cdchn: getsem");
    sjy_putsem();
    return (CAM_S_GETSEM_FAIL);
  }

  /* open the scsi device file */
  if ((fd = open(filename,O_RDWR) ) == -1) {
      fprintf(stderr, "CDCHN Error opening %s\n",filename);
      perror("CDCHN");
      sjy_putsem();
      return (CAM_S_DEV_OPEN_FAIL);
      }

  /* release the semaphore */
  if(sjy_putsem() != 0) {
    perror("cdchn: putsem");
    return (CAM_S_PUTSEM_FAIL);
  }

#ifdef DEBUG
  fprintf(stderr, "cdchn: filename = %s  fd = %d\n", filename, fd);
#endif

  sjy_controller[idx].scsi_id  = ID(branch);
  sjy_controller[idx].scsi_bus = BUS(branch);
  sjy_controller[idx].serial   = 1;  /* assume serial */
  sjy_controller[idx].fd       = fd;

  /* Send two Test Unit Ready Commands to Jorway */
  status = sjy_tur(branch);
  if (status) {                        /* 0=success */
    fprintf(stderr, "CDCHN Error executing Test Unit Ready\n");
    perror("cdchn");
    return (CAM_S_DEV_OPEN_FAIL);
  }

  return (CAM_S_SUCCESS);
}
/* -----------------------------------------------------------------------   */
/* ************************************************************************* */
/* ===================================================================== 
**
** SUBROUTINE	CDCHN_
**
** DESCRIPTION
**	Subroutine to initialize CAMAC environment
**	It opens the vme device file and establish
**	the mapping between the object file and the user application.
**
**	This stub is to make the INICAM call possible from a FORTRAN program
**
** DEVELOPERS
**      Margherita Vittone
**      Fermilab Data Acquisition Group
**      Batavia, Il 60510, U.S.A.
**
** MODIFICATIONS
**         Date       Initials  Description
**      21-Aug-1991	MVW     Creation
**
**  ======================================================================== */
/*									     */
void cdchn_ (int	*branch,
              int	*channel,
              int	*route)
{
    ieee_status = cdchn(*branch,*channel,*route);
}

/* -----------------------------------------------------------------------   */

/* 
** Read file "filename" on stream g_rfp and return
** results in buffer charBuff.
*/

int readFile(FILE *g_rfp, char *filename, char charBuff[], int charBuffsize)
{
  int status;

                         
  if (fgets(charBuff, charBuffsize, g_rfp) != NULL) {
    strtok(charBuff, "\n");
    status = 0;
  } else {   /* error or end of file found and no characters read */
    status = EOF;
  }
      
  return(status);
}

/* ************************************************************************* */
/* ===================================================================== 
**
** SUBROUTINE	sjy_getdev
**
** DESCRIPTION
**	Find the Jorway devices in /proc/scsi/scsi.This only scans the 
**      SCSI devices found in /proc/scsi/scsi. This routine assumes links
**      from /dev/sg0 -> sga, /dev/sg1 -> sgb ....  
**      Return the device matching the SCSI Id and SCSI bus encoded in
**      the branch argument.
**	
**
** DEVELOPERS
**      Dave Slimmer
**      Fermilab Data Acquisition Group
**      Batavia, Il 60510, U.S.A.
**
** MODIFICATIONS
**         Date       Initials  Description
**      25-Aug-1991	DAS     Creation
**      25-Feb-2003     DAS     added Thomas Hadig fix to sscanf for long 
**                              vendor and type names
**                             
**  ======================================================================== */
/*									     */

static char *devtble[] =   {"/dev/sg0",   /* scsi generic devices */
			    "/dev/sgb",
			    "/dev/sgc",
			    "/dev/sgd",
			    "/dev/sge",
			    "/dev/sgf",
			    "/dev/sgh",
			    "/dev/sgg",
			    "/dev/sgh"};

int sjy_getdev (char *dev, int branch)
{
 int status = 0;
 char charBuff[120];        /* file line buffer */

 FILE *g_rfp;               /* /proc/scsi/scsi file descriptor */
 char filename[32]={"/proc/scsi/scsi"};  /* file describing SCSI devices attached to system */
 int bus;                   /* SCSI bus */ 
 int id;                    /* SCSI id */
 char vendor[10];           /* SCSI device vendor */
 char type[10];             /* SCSI device type */
 int found = 0;             /* Jorway found flag */
 int i = 0;                 /* devtble index */
 int devtblesz = 9;         /* size of devtble */

 
 /* open /proc/scsi/scsi */	
 g_rfp = fopen(filename, "r");
 if(g_rfp == NULL)
   {
     printf("ScanSCSIDevices:  Cannot open %s", filename);
     return(CAM_S_DEV_OPEN_FAIL);
   }

 /* read each line of filename */
 while((status == 0) && (i < devtblesz))
   {
     
     status = readFile(g_rfp, filename, charBuff, sizeof(charBuff));
     if(status == EOF) {    /* end of file */
      
	 if (!found) status = CAM_S_DEV_OPEN_FAIL;
	 break;

     } else {    /* parse the file line */

	 /* parse the Host line */
	 if (strncmp(charBuff, "Host", 4) == 0) {
	   sscanf(&charBuff[10],"%d",&bus);
	   sscanf(&charBuff[29],"%d",&id);
	 }
	 
	 /* parse the Vendor line */
	 if (strncmp(&charBuff[2], "Vendor", 6) == 0) {
	   sscanf(&charBuff[10],"%9s", vendor);
	   
	   if (strcmp(vendor, "JORWAY") == 0) {
	     if ( (id == ID(branch)) && (bus == BUS(branch)) ) {
	       /* found our device by matching scsi id */  
	       strcpy(dev, devtble[i]);
	       found = 1;
	       break;
	     } else { /* JORWAY found, but not correct ID/BUS */
              /* TH010923 Thomas Hadig hadig@slac.stanford.edu: 
               * this fixes the libraries behaviour if you have more than one
               * JORWAY SCSI-CAMAC interface in your PC 
               */ 
              i++; 
             }
	   } else {  /* SCSI device found, but not a Jorway */
	     i++;
	   }
	 }
	 
         /* parse the Type line */
	 if (strncmp(&charBuff[2], "Type", 4) == 0) {
	   sscanf(&charBuff[10],"%9s", type);
#ifdef DEBUG
           if (strcmp(type, "Unknown") || (strcmp(type, "Processor")) {
	     printf("Found type %s", type);
	   }
#endif	
	 }

       } /* if status == 0 */
   } /* while */

 fclose(g_rfp);
 if (found) status = CAM_S_SUCCESS;
 return(status);

}




