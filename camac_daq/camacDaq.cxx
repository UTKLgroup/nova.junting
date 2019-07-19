/*********************************************************************
A-camacDaq
this program will take data from the CAMAC and save it in the original
format which has line breaks stating the event number.  this uses up
about twice as much space so i've edited this program to be BcamacDaq
and does not include that extra text.  because of the difference in the
data file the A(macro) series of macros are for the AcamacDaq and the
B(macro) series of macros are for the BcamacDaq!
*********************************************************************/

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <signal.h>
#include <sys/time.h>
#include "ieee_fun_types.h"
#include "configLib.h"
#include "keyValuePair.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TFile.h"
#include "TF1.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TApplication.h"


typedef enum {FALSE, TRUE} bool_t ;

#define NCHAN_2249 12
#define NCHAN_2228 8

//static u_int32_t ext2249[NCHAN_2249];
int ext2249[NCHAN_2249];

//static u_int32_t ext2228[NCHAN_2228];
int ext2228[NCHAN_2228];

//static u_int32_t outreg_ext ;
int outreg_ext ;

static int branch = 1; //this must be 9 but it's NOT the slot# which is specified in the nemoDaq.config!!!
static int crate = 1;
static bool_t runActive;
static FILE* outputFile;


static void timerHandler(int signal)
{
   runActive = FALSE;
}


int open_camac (void)
{
  /* open the SCSI device */
  if((cdchn (branch, 1, 0) &1) != 1)
    {
      perror ("cdchn error (2)") ;
      return (2) ;
    }
  
  /* set the controller type */
  ccctype(branch,0,1); /* parallel - Jor 73A is considered parallel */
  
  return (0) ;
}


void caminit (void)
{
  int ext;
  int slot;
  int subaddr;
  
  /* We're dealing with the crate so N and A are not significant */
  slot = 1;
  subaddr = 0;
  cdreg (&ext, branch, crate, slot, subaddr);
  cccc (ext);
  cccz (ext);
  ccci (ext, FALSE);
}


int adcRead(u_int16_t* buffer)
{
  int subaddr;
  int qresponse;
  
  for (subaddr = 0; subaddr<NCHAN_2249; subaddr++) 
    {
      cssa (0, ext2249[subaddr], &buffer[subaddr], &qresponse);
    }
  return(0);
}


void adcClear()
{
   u_int16_t dwords[2];
   int qresponse;
   cssa (9, ext2249[0], dwords, &qresponse);
}


int tdcRead(u_int16_t* buffer)
{
  int subaddr;
  int qresponse;
  
  for (subaddr = 0; subaddr<NCHAN_2228; subaddr++) 
    {
      cssa (0, ext2228[subaddr], &buffer[subaddr], &qresponse);
    }
  return(0);
}


void tdcClear()
{
  u_int16_t dwords[2];
  int qresponse;
  cssa (9, ext2228[0], dwords, &qresponse);
}


int waitForLAM(int ext)
{
  int npolls;
  int qresponse;
  int delay;
  u_int16_t dwords[2];
  volatile int dummy;
  
  npolls = 0;
  do {
    cssa (8, ext, dwords, &qresponse) ;
    if (!qresponse)
      {
	npolls++ ;
	if (npolls < 2)
	  {
            for (delay = 0 ; delay < 150000 ; delay++) dummy = delay ;
	  }
	else
	  {
            usleep (1) ;
	  }
      }
  } while (runActive && !qresponse) ;
  
  return(qresponse);
}


void removeTriggerInhibit (u_int16_t or_data)
{
  int or_q;
  
#ifdef TRACE
  printf ("In remove_inhibit: calling cssa\n") ;
#endif
  cssa (17, outreg_ext, &or_data, &or_q);
}

int openOutput(char*outputPath, int runNumber)
{
  char fname[FILENAME_MAX] ;
  int status ;
  
  status = 0 ;
  snprintf (fname, sizeof(fname), "%s/run%0d.dat", outputPath, runNumber) ;
  outputFile = fopen (fname, "w") ;
  if (outputFile == NULL)
    {
      perror ("openOutput") ;
      status = 1 ;
    }
  return(status);
}


void trapSignal(int signal, void (*handler)(int))
{
  struct sigaction action ;
  
  /* Register signal handler */
  action.sa_handler = handler ;
  sigemptyset (&action.sa_mask) ;
  action.sa_flags = 0 ;
  sigaction (signal, &action, NULL) ;
}


void setupTimer(int seconds)
{
  struct itimerval interval;
  
  /* Setup the interval timer */
  interval.it_interval.tv_sec = 0;
  interval.it_interval.tv_usec = 0;
  
  interval.it_value.tv_sec = seconds;
  interval.it_value.tv_usec = 0;
  
  setitimer(ITIMER_REAL, &interval, 0);
  
}



int main(int argc, char* argv[])
{
    u_int16_t adcValues[12];
    u_int16_t tdcValues[8];

    //  Int_t adcValues[12];
    //  Int_t tdcValues[8];

    int channel;
    int slot2249;
    int slot2228;
    int slotOutput;
    u_int16_t or_data;
    
    //  bool_t useADC;
    //  bool_t useTDC;
    int useADC;
    int useTDC;
    int runTime;
    int maxEvents;
    int eventNumber;
    int runNumber;
    //int rnStream;  //was getting a warning that this was an unused variable
    char* outputPath;

    time_t start = time(NULL);
    time_t end;
    int elapsed;

    TFile *file = new TFile("acquisition.root","RECREATE");
    TApplication* rootapp = new TApplication("example",&argc, argv);

    //create the ROOT tree to store the data
    char branchname[20];

    //    sprintf(branchname,"Run_%d",1);
    sprintf(branchname,"ADC_VALUES");

    TTree *EventTree = new TTree(branchname, branchname);


    TBranch *ADC_Branch_0;
    TBranch *ADC_Branch_1;
    TBranch *ADC_Branch_2;
    TBranch *ADC_Branch_3;
    TBranch *ADC_Branch_4;
    TBranch *ADC_Branch_5;
    TBranch *ADC_Branch_6;
    TBranch *ADC_Branch_7;
    TBranch *ADC_Branch_8;
    TBranch *ADC_Branch_9;
    TBranch *ADC_Branch_10;
    TBranch *ADC_Branch_11;

    Int_t adc0, adc1, adc2, adc3, adc4, adc5, adc6, adc7, adc8, adc9, adc10, adc11;

    ADC_Branch_0 = EventTree->Branch("ADC_Runs_0",&adc0,"adc0/I");
    ADC_Branch_1 = EventTree->Branch("ADC_Runs_1",&adc1,"adc1/I");
    ADC_Branch_2 = EventTree->Branch("ADC_Runs_2",&adc2,"adc2/I");
    ADC_Branch_3 = EventTree->Branch("ADC_Runs_3",&adc3,"adc3/I");
    ADC_Branch_4 = EventTree->Branch("ADC_Runs_4",&adc4,"adc4/I");
    ADC_Branch_5 = EventTree->Branch("ADC_Runs_5",&adc5,"adc5/I");
    ADC_Branch_6 = EventTree->Branch("ADC_Runs_6",&adc6,"adc6/I");
    ADC_Branch_7 = EventTree->Branch("ADC_Runs_7",&adc7,"adc7/I");
    ADC_Branch_8 = EventTree->Branch("ADC_Runs_8",&adc8,"adc8/I");
    ADC_Branch_9 = EventTree->Branch("ADC_Runs_9",&adc9,"adc9/I");
    ADC_Branch_10 = EventTree->Branch("ADC_Runs_10",&adc10,"adc10/I");
    ADC_Branch_11 = EventTree->Branch("ADC_Runs_11",&adc11,"adc11/I");

    TString histo = "Histogram";
    TH1F *hadcValues[12];
    //    hadcValues[0]  = new TH1F("ADC_0",  "ADC_0", 100, 1550, 1649);
    hadcValues[0]  = new TH1F("ADC_0",  "ADC_0", 2048, 0, 2047);
    hadcValues[1]  = new TH1F("ADC_1",  "ADC_1", 1024, 0, 1023);
    hadcValues[2]  = new TH1F("ADC_2",  "ADC_2", 1024, 0, 1023);
    hadcValues[3]  = new TH1F("ADC_3",  "ADC_3", 1024, 0, 1023);
    hadcValues[4]  = new TH1F("ADC_4",  "ADC_4", 1024, 0, 1023);
    hadcValues[5]  = new TH1F("ADC_5",  "ADC_5", 1024, 0, 1023);
    hadcValues[6]  = new TH1F("ADC_6",  "ADC_6", 1024, 0, 1023);
    hadcValues[7]  = new TH1F("ADC_7",  "ADC_7", 1024, 0, 1023);
    hadcValues[8]  = new TH1F("ADC_8",  "ADC_8", 1024, 0, 1023);
    hadcValues[9]  = new TH1F("ADC_9",  "ADC_9", 1024, 0, 1023);
    hadcValues[10] = new TH1F("ADC_10", "ADC_10", 1024, 0, 1023);
    hadcValues[11] = new TH1F("ADC_11", "ADC_11", 1024, 0, 1023);
    TH1F *sum = new TH1F("sum", "sum", 1024, 0, 1023); 

    FILE* rnFile;

    TCanvas *can1 = new TCanvas("can1","Canvas", 200, 10, 500, 500);
    can1->Draw();

    /*
    TCanvas *can2 = new TCanvas("can2","Canvas", 200, 10, 500, 500);
    can2->Draw();
    TCanvas *can3 = new TCanvas("can3","Canvas", 200, 10, 500, 500);
    can3->Draw();
    TCanvas *cansum = new TCanvas("cansum","Canvas", 200, 10, 500, 500);
    cansum->Draw();
    */

    configLoad("runNumber.config","run");
    runNumber = kvpGetInt("runNumber", 0);
    runNumber++;
    rnFile = fopen("runNumber.config", "w");
    fprintf(rnFile,"<run>\nrunNumber#S=%d;\n</run>\n", runNumber);
    fclose(rnFile);
    
    /*** Get configuration from file into kvp buffer ***/
    configLoad("nemoDaq.config","global");
    useADC = kvpGetInt("useADC", 0);
    if (useADC == 1) 
      
      {
	  slot2249 = kvpGetInt("slot2249", 0);
	  if (slot2249 == 0) 
	      {
		fprintf(stderr, "ERROR: slot2249 not defined!\n");
		return(EXIT_FAILURE);
	      }
	  for (channel = 0; channel<NCHAN_2249; channel++) 
	      {
		  cdreg (&ext2249[channel], branch, crate, slot2249, channel);
	      }
      }
    useTDC = kvpGetInt("useTDC", 0);
    if (useTDC) 
        {
	    slot2228 = kvpGetInt("slot2228", 0);
	    if (slot2228 == 0) 
	        {
		  fprintf(stderr, "ERROR: slot2228 not defined!\n");
		  return(EXIT_FAILURE);
		}
	    
	    for (channel = 0; channel<NCHAN_2228; channel++) 
	        {
		    cdreg (&ext2228[channel], branch, crate, slot2228, channel);
		}
	}
    
    slotOutput = kvpGetInt("slotOutput", 0);
    if (slotOutput == 0) 
        {
	    fprintf(stderr, "ERROR: slotOutput not defined!\n");
	    return(EXIT_FAILURE);
	}
    cdreg (&outreg_ext, branch, crate, slotOutput, 0);
    or_data = kvpGetInt("valueOutput", 1);
    
    runTime = kvpGetInt("runTime", 0);
    
    maxEvents = kvpGetInt("maxEvents",0);
    
    outputPath = kvpGetString("outputPath");
    if (outputPath != NULL) 
        {
	  openOutput(outputPath, runNumber);
	}
    else 
        {
	    outputFile = stdout;
	}
    
    open_camac();
    caminit();
    
    runActive = TRUE;
    eventNumber = 0;
    if (runTime != 0) 
        {
	    trapSignal(SIGALRM, timerHandler);
	    setupTimer(runTime);
	}
    trapSignal(SIGINT, timerHandler);
    while (runActive) 
        {
	    removeTriggerInhibit(or_data);
	    
	    if (useADC  == 1) 
	        {
		    waitForLAM(ext2249[0]);
		    adcRead(adcValues);
		    adcClear();
		}
	    
	    if (useTDC) 
	        {
		    waitForLAM(ext2228[0]);
		    tdcRead(tdcValues);
		    tdcClear();
		}
	    if (runActive) 
	        {
		    fprintf(outputFile, "\nEvent %d", eventNumber);  //don't want this in datfile
		    if (useADC == 1) {
		        fprintf(outputFile, "\nADC:");  //don't want this in datfile
			if (eventNumber%10 == 0){
			    //     std::cout << "Event: " << eventNumber << std::endl;

			    end = time(NULL);
			    elapsed = (end - start);
			    std::cout << "Event: " << eventNumber 
				      << "     Rate: " << (double)eventNumber/elapsed 
				      << " Hz" << std::endl;
			    /*
			      ofstream timeout;
			      timeout.open("Runtime.out",std::ios::app);
			      timeout << "Run " << aRun->GetRunID() << ": " 
			      << elapsed/3600 << ":" 
			      << (elapsed%3600)/60 << ":" 
			      << ((elapsed%3600)%60) << G4endl;
			      timeout.close();
			    */
			    
			}


			// subtract pedestal for ONLY channel 0 !   must do the same for other channels!
			for (channel = 0; channel < 1; channel++) { 
			  hadcValues[channel]->Fill((adcValues[channel]) - 198 );  // Check that this is correct!
			}

			for (channel = 1; channel<NCHAN_2249; channel++) { 
			    fprintf(outputFile, "%i ", adcValues[channel]);
			    //printf("%i ",adcValues[channel]);  //print to screen
			    //  hadcValues[channel]->Fill(adcValues[channel]);
			    
		       

			    hadcValues[channel]->Fill((adcValues[channel]) );
			    
			    
			    adc0 = adcValues[0];
			    adc1 = adcValues[1];
			    adc2 = adcValues[2];
			    adc3 = adcValues[3];
			    adc4 = adcValues[4];
			    adc5 = adcValues[5];
			    adc6 = adcValues[6];
			    adc7 = adcValues[7];
			    adc8 = adcValues[8];
			    adc9 = adcValues[9];
			    adc10 = adcValues[10];
			    adc11 = adcValues[11];
			    
			}
			sum->Fill((adcValues[0]-16) + (adcValues[1]-33));
			EventTree->Fill();
			can1->cd();


			/*		
			//new stuff from Benton for fitting
			if (eventNumber%10 == 0){
			

			  hadcValues[0]->Fit("landau","","", 1000, 2048);
			  TF1 *gausfit = hadcValues[0]->GetFunction("landau");
			  
			  Double_t mu = 0;
			  Double_t sigma = 0;
			  Double_t mu_error = 0;
			  Double_t sigma_error = 0;
			  


			  mu = gausfit->GetParameter(1);
			  sigma = gausfit->GetParameter(2);
			  mu_error = gausfit->GetParError(1);
			  sigma_error = gausfit->GetParError(2);			  

			    std::cout << std::endl;
			    std::cout << "Mean = "  << mu    << std::endl;
			    std::cout << "Sigma = " << sigma << std::endl;
			  
			//			  hadcValues[0]->Fit("gaus","","", mu-5*sigma, mu+9*sigma);
			  
			// std::cout << std::endl;
			//  std::cout << "FINAL FIT RESULTS" << std::endl;
			//  std::cout << "Mean = "  << mu    << " =/- " << mu_error << std::endl;
			//  std::cout << "Sigma = " << sigma << " +/- " << sigma_error << std::endl;

			//			}


			// end of Benton's shit
			
		        */

			hadcValues[0]->Draw();
			can1->Modified();
			can1->Update();
			/*
			can2->cd();
			hadcValues[1]->Draw();	
			can2->Modified();
			can2->Update();

			can3->cd();
			hadcValues[2]->Draw();	
			can3->Modified();
			can3->Update();

			cansum->cd();
			sum->Draw();
			cansum->Modified();
			cansum->Update();
			*/

			fprintf(outputFile,"\n");
			//printf("\n");  //print to screen
		    }
		    //		    EventTree->Fill();

		    if (useTDC){ 
		      
			    fprintf(outputFile, "\nTDC:");
			    for (channel = 0; channel<NCHAN_2228; channel++) 
			        {
				    fprintf(outputFile, " %i", tdcValues[channel]);
				}
			}
		    eventNumber++;
		    if ((maxEvents != 0) && (eventNumber >= maxEvents)) 
		        {
			  runActive = FALSE;
			}
		}
	}
    for (int i = 0; i < 12; i++){
      hadcValues[i]->Write();
    }

    // apparently not required according to Fons    rootapp->Run();

    EventTree->Print();
    EventTree->Write();
    sum->Write();
    fclose(outputFile);
    file->Close();

    std::cout << "DONE!  " << std::endl;
    end = time(NULL);
    elapsed = (end - start);
    std::cout << "Run Completed in " << elapsed/3600 << ":"
	      << (elapsed%3600)/60 << ":" << ((elapsed%3600)%60) << std::endl;

    return 0;
}

