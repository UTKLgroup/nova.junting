#include <string>
#include <fstream>
#include <iostream>

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

  NtpMCEventRecord* mcrec = 0;
  tree->SetBranchAddress("gmcrec", &mcrec);
  int nev = (gOptNEvt > 0) ?
    TMath::Min(gOptNEvt, (int)tree->GetEntries()) :
    (int) tree->GetEntries();

  for(int i = 0; i < nev; i++) {
    tree->GetEntry(i);
    EventRecord & event = *(mcrec->event);
    GHepParticle* p = 0;
    TIter event_iter(&event);

    vector<vector<double>> particle_infos;
    const int info_count = 6;

    while((p=dynamic_cast<GHepParticle *>(event_iter.Next()))) {
      if (p->Status() == kIStStableFinalState ) {
        double infos[info_count] = {
          (double) p->Pdg(),
          p->Px(),
          p->Py(),
          p->Pz(),
          p->Energy(),
          p->Mass()
        };
        particle_infos.push_back(vector<double>(infos, infos + info_count));
      }
    } // end loop over particles

    output << TString::Format("0 %lu\n", particle_infos.size());
    for (unsigned int j = 0; j < particle_infos.size(); j++) {
      output << TString::Format("1 %.0f 0 0 0 0 %f %f %f %f %f 0. 0. 3000. 225000\n",
                                particle_infos[j][0],
                                particle_infos[j][1],
                                particle_infos[j][2],
                                particle_infos[j][3],
                                particle_infos[j][4],
                                particle_infos[j][5]);
    }

    mcrec->Clear();
  } //end loop over events

  output.close();
  file.Close();
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
