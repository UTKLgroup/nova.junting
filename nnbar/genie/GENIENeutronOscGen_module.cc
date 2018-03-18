////////////////////////////////////////////////////////////////////////
/// \brief  GENIE neutron oscillation event generator
/// \author junting@utexas.edu
/// \date
////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <unistd.h>

// ROOT includes
#include "TStopwatch.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include "TSystem.h"

// Framework includes
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/SubRun.h"
#include "fhiclcpp/ParameterSet.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

// NOvA includes
// #include "EventGeneratorBase/GENIE/GENIEHelper.h"
#include "nutools/EventGeneratorBase/GENIE/GENIEHelper.h"
#include "Geometry/Geometry.h"
// #include "SimulationBase/GTruth.h"
// #include "SimulationBase/MCFlux.h"
// #include "SimulationBase/MCTruth.h"
#include "nusimdata/SimulationBase/GTruth.h"
#include "nusimdata/SimulationBase/MCFlux.h"
#include "nusimdata/SimulationBase/MCTruth.h"
#include "SummaryData/POTSum.h"
#include "SummaryData/SpillData.h"
#include "SummaryData/RunData.h"
#include "SummaryData/SubRunData.h"
#include "Utilities/AssociationUtil.h"

#include "dk2nu/tree/dk2nu.h"
#include "dk2nu/tree/NuChoice.h"
#include "dk2nu/genie/GDk2NuFlux.h"

#include "EVGCore/EventRecord.h"
#include "EVGCore/EventRecordVisitorI.h"
#include "Algorithm/AlgFactory.h"
#include "Numerical/RandomGen.h"
#include "Ntuple/NtpWriter.h"
#include "Messenger/Messenger.h"
#include "GHEP/GHepParticle.h"
#include "NeutronOsc/NeutronOscMode.h"
#include "NeutronOsc/NeutronOscUtils.h"

namespace evgen {

  class GENIENeutronOscGen : public art::EDProducer {

  public:

    explicit GENIENeutronOscGen(fhicl::ParameterSet const &pset);
    virtual ~GENIENeutronOscGen();

    void produce(art::Event& evt);
    void beginJob();
    void beginRun(art::Run &run);
    void endSubRun(art::SubRun &sr);
    const genie::EventRecordVisitorI* NeutronOscGenerator(void);
    int selectAnnihilationMode(int pdg_code);
    void fillMCTruth(const genie::EventRecord* record, simb::MCTruth &truth);
    void setRandomEventVertexPosition(genie::EventRecord* record);

  private:

    TStopwatch fStopwatch;                 ///< keep track of how long it takes to run the job
    int fCycle;                            ///< cycle number in the MC generation
    int target;                            ///< pdg code of the target
    genie::NeutronOscMode_t gOptDecayMode; ///< neutron oscillation mode
    bool randomVertexPosition;
    const genie::EventRecordVisitorI* mcgen;
  };
};

namespace evgen {

  //___________________________________________________________________________
  GENIENeutronOscGen::GENIENeutronOscGen(fhicl::ParameterSet const& pset)
    : fCycle(pset.get<int>("Cycle", 0))
    , target(pset.get<int>("Target", 1000060120))
    , randomVertexPosition(pset.get<bool>("RandomVertexPosition", false))
  {
    fStopwatch.Start();

    gOptDecayMode = (genie::NeutronOscMode_t) pset.get<int>("AnnihilationMode", 0);
    bool valid_mode = genie::utils::neutron_osc::IsValidMode(gOptDecayMode);
    if (!valid_mode) {
      mf::LogError("GENIENeutronOscGen") << "You need to specify a valid annihilation mode (0-16)";
      exit(0);
    }

    mcgen = NeutronOscGenerator();

    produces<std::vector<simb::MCTruth>>();
    produces<sumdata::SubRunData, art::InSubRun>();
    produces<sumdata::RunData, art::InRun>();
  }

  //___________________________________________________________________________
  GENIENeutronOscGen::~GENIENeutronOscGen()
  {
    delete mcgen;

    fStopwatch.Stop();
    mf::LogInfo("GENIENeutronOscGen") << "real time to produce file: "
                                      << fStopwatch.RealTime();
  }

  //___________________________________________________________________________
  void GENIENeutronOscGen::beginJob()
  {
  }

  //___________________________________________________________________________
  void GENIENeutronOscGen::beginRun(art::Run& run)
  {
    art::ServiceHandle<geo::Geometry> geo;

    std::unique_ptr<sumdata::RunData>
      runcol(new sumdata::RunData(geo->DetId(),
                                  geo->FileBaseName(),
                                  geo->ExtractGDML()));

    run.put(std::move(runcol));

    return;
  }

  //___________________________________________________________________________
  void GENIENeutronOscGen::endSubRun(art::SubRun &sr)
  {
    art::ServiceHandle<geo::Geometry> geo;
    std::unique_ptr< sumdata::SubRunData > sd(new sumdata::SubRunData(fCycle));
    sr.put(std::move(sd));
  }

  //___________________________________________________________________________
  void GENIENeutronOscGen::produce(art::Event& evt)
  {
    std::unique_ptr< std::vector<simb::MCTruth> > truthcol(new std::vector<simb::MCTruth>);
    simb::MCTruth truth;

    int decay = selectAnnihilationMode(target);
    genie::Interaction* interaction = genie::Interaction::NOsc(target, decay);
    genie::EventRecord* event = new genie::EventRecord;
    event->AttachSummary(interaction);
    mcgen->ProcessEventRecord(event);
    // mf::LogInfo("GENIENeutronOscGen") << "Generated event: " << *event;

    event->SetVertex(0., 0., 30., 225.e-6); // in si unit, m and s
    if (randomVertexPosition) {
      setRandomEventVertexPosition(event);
    }

    fillMCTruth(event, truth);
    truthcol->push_back(truth);

    evt.put(std::move(truthcol));
  }

  //___________________________________________________________________________
  void GENIENeutronOscGen::setRandomEventVertexPosition(genie::EventRecord* record)
  {
    genie::RandomGen* rnd = genie::RandomGen::Instance();
    rnd->SetSeed(0);
    double p = rnd->RndNum().Rndm();

    art::ServiceHandle<geo::Geometry> geo;
    double half_width = geo->DetHalfWidth() / 100.;   // m
    double half_height = geo->DetHalfHeight() / 100.; // m
    double length = geo->DetLength() / 100.;          // m

    double random_x = (p - 0.5) * half_width;  // m
    double random_y = (p - 0.5) * half_height; // m
    double random_z = p * length;              // m

    record->SetVertex(random_x, random_y, random_z, record->Vertex()->T());
  }

  //___________________________________________________________________________
  const genie::EventRecordVisitorI* GENIENeutronOscGen::NeutronOscGenerator(void)
  {
    string sname   = "genie::EventGenerator";
    string sconfig = "NeutronOsc";
    genie::AlgFactory* algf = genie::AlgFactory::Instance();
    const genie::EventRecordVisitorI* mcgen = dynamic_cast<const genie::EventRecordVisitorI *> (algf->GetAlgorithm(sname,sconfig));

    if (!mcgen) {
      mf::LogError("GENIENeutronOscGen") << "Couldn't instantiate the neutron oscillation generator";
      genie::gAbortingInErr = true;
      exit(1);
    }

    return mcgen;
  }

  //___________________________________________________________________________
  int GENIENeutronOscGen::selectAnnihilationMode(int pdg_code)
  {
    if (gOptDecayMode == genie::kNORandom) {
      int mode;

      std::string pdg_string = std::to_string(static_cast<long long>(pdg_code));
      if (pdg_string.size() != 10) {
        mf::LogError("GENIENeutronOscGen") << "Expecting PDG code to be a 10-digit integer; instead, it's the following: " << pdg_string;
        genie::gAbortingInErr = true;
        exit(1);
      }

      int n_nucleons = std::stoi(pdg_string.substr(6,3)) - 1;
      int n_protons  = std::stoi(pdg_string.substr(3,3));

      double proton_frac  = ((double)n_protons) / ((double)n_nucleons);
      double neutron_frac = 1 - proton_frac;

      const int n_modes = 16;
      double br [n_modes] = { 0.010, 0.080, 0.100, 0.220,
                              0.360, 0.160, 0.070, 0.020,
                              0.015, 0.065, 0.110, 0.280,
                              0.070, 0.240, 0.100, 0.100 };

      for (int i = 0; i < n_modes; i++) {
        if (i < 7)
          br[i] *= proton_frac;
        else
          br[i] *= neutron_frac;
      }

      genie::RandomGen* rnd = genie::RandomGen::Instance();
      rnd->SetSeed(0);
      double p = rnd->RndNum().Rndm();

      double threshold = 0;
      for (int i = 0; i < n_modes; i++) {
        threshold += br[i];
        if (p < threshold) {
          mode = i + 1;
          return mode;
        }
      }

      mf::LogError("GENIENeutronOscGen") << "Random selection of final state failed!";

      genie::gAbortingInErr = true;
      exit(1);
    }

    else {
      int mode = (int) gOptDecayMode;
      return mode;
    }
  }

  //___________________________________________________________________________
  void GENIENeutronOscGen::fillMCTruth(const genie::EventRecord* record, simb::MCTruth &truth)
  {
    TLorentzVector* vertex = record->Vertex();
    TIter partitr(record);
    genie::GHepParticle* part = 0;
    int trackid = 0;
    std::string primary("primary");

    while( (part = dynamic_cast<genie::GHepParticle*>(partitr.Next())) ) {
      if (part->Status() != genie::kIStStableFinalState ) {
        continue;
      }

      simb::MCParticle tpart(trackid,
                             part->Pdg(),
                             primary,
                             part->FirstMother(),
                             part->Mass(),
                             part->Status());

      TLorentzVector* vertex = record->Vertex();
      double vtx[4] = {
        100. * (part->Vx() * 1.e-15 + vertex->X()), // cm
        100. * (part->Vy() * 1.e-15 + vertex->Y()), // cm
        100. * (part->Vz() * 1.e-15 + vertex->Z()), // cm
        part->Vt() + vertex->T() * 1.e9             // ns
      };

      TLorentzVector position(vtx[0], vtx[1], vtx[2], vtx[3]);
      TLorentzVector momentum(part->Px(), part->Py(), part->Pz(), part->E());
      tpart.AddTrajectoryPoint(position, momentum);
      truth.Add(tpart);
      trackid++;
    }
  }

}

namespace evgen { DEFINE_ART_MODULE(GENIENeutronOscGen); }
