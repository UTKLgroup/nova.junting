#include <string>
#include <fstream>

#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TIterator.h>

#include "EVGCore/EventRecord.h"
#include "GHEP/GHepParticle.h"
#include "Ntuple/NtpMCFormat.h"
#include "Ntuple/NtpMCTreeHeader.h"
#include "Ntuple/NtpMCEventRecord.h"
#include "Messenger/Messenger.h"
#include "PDG/PDGCodes.h"
#include "Utils/CmdLnArgParser.h"

using std::string;
using namespace genie;

void GetCommandLineArgs (int argc, char ** argv);

int    gOptNEvt;
string gOptInpFilename;

int main(int argc, char ** argv)
{
  GetCommandLineArgs (argc, argv);

  TTree* tree = 0;
  NtpMCTreeHeader* thdr = 0;
  TFile file(gOptInpFilename.c_str(), "READ");
  tree = dynamic_cast <TTree*> (file.Get("gtree"));
  thdr = dynamic_cast <NtpMCTreeHeader*> (file.Get("header"));
  if(!tree) {
    return 1;
  }

  ofstream output;
  output.open(TString::Format("%s.txt", gOptInpFilename.c_str()));
  output << "Writing this to a file.\n";

  NtpMCEventRecord* mcrec = 0;
  tree->SetBranchAddress("gmcrec", &mcrec);
  int nev = (gOptNEvt > 0) ? TMath::Min(gOptNEvt, (int)tree->GetEntries()) : (int) tree->GetEntries();
  for(int i = 0; i < nev; i++) {
    tree->GetEntry(i);
    EventRecord & event = *(mcrec->event);
    GHepParticle* p = 0;
    TIter event_iter(&event);

    vector<vector<double>> particles;
    while((p=dynamic_cast<GHepParticle *>(event_iter.Next()))) {
       if (p->Status() == kIStStableFinalState ) {
	  // if (p->Pdg() == kPdgPi0 ||
	  //     p->Pdg() == kPdgPiP ||
	  //     p->Pdg() == kPdgPiM) {
          //   LOG("myAnalysis", pNOTICE) << "Got a : " << p->Name() << " with E = " << p->E() << " GeV";
          // }
       }
    } // end loop over particles
    mcrec->Clear();
  } //end loop over events

  file.Close();
  output.close();
  LOG("myAnalysis", pNOTICE)  << "Done!";
  return 0;
}

void GetCommandLineArgs(int argc, char** argv)
{
  LOG("myAnalysis", pINFO) << "Parsing commad line arguments";

  CmdLnArgParser parser(argc,argv);

  // get GENIE event sample
  if( parser.OptionExists('f') ) {
    LOG("myAnalysis", pINFO)
       << "Reading event sample filename";
    gOptInpFilename = parser.ArgAsString('f');
  } else {
    LOG("myAnalysis", pFATAL)
        << "Unspecified input filename - Exiting";
    exit(1);
  }

  // number of events to analyse
  if( parser.OptionExists('n') ) {
    LOG("myAnalysis", pINFO)
      << "Reading number of events to analyze";
    gOptNEvt = parser.ArgAsInt('n');
  } else {
    LOG("myAnalysis", pINFO)
      << "Unspecified number of events to analyze - Use all";
    gOptNEvt = -1;
  }
}
