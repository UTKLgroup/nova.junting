#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;
string CodetoName (int a);

void fraction(string addr="total.root") {
  gStyle->SetOptFit(0000);
  gStyle->SetStatFontSize(0.05);
  gStyle->SetOptStat("rme");
  gStyle->SetStatW(0.3);
  TGaxis::SetMaxDigits(3);

  TCanvas *c=new TCanvas("c","c",800,500);
  string dir="/nova/data/users/junting/g4bl/";
  rootname = dir + addr;
  cout<<"reading "<<rootname<<endl;
  TFile* f = new TFile(rootname.c_str());

  f->cd("VirtualDetector");
  TTree* nt = (TTree*) gDirectory->Get("Argu");

  TH1I* h = new TH1I("h", "h", 6000, -3000, 3000);
  nt->Draw("PDGid >> h");
  h->Draw();
  h->SetTitle("Particles at Argoneut");
  h->GetXaxis()->SetLabelSize(0.05);
  h->GetXaxis()->SetTitleSize(0.05);
  h->GetYaxis()->SetLabelSize(0.05);
  h->GetYaxis()->SetTitleSize(0.05);
  h->GetXaxis()->SetTitle("PDG code");
  h->GetYaxis()->SetTitle("Number of Particles");

  TAxis* axis = h->GetXaxis();
  vector < int > ptcl;
  vector < double > nptcl;
  vector < string > pname;
  for(int i = 0; i < h->GetSize(); i++){
    if (h->GetBinContent(i) != 0){
      ptcl.push_back(i-3001);
      nptcl.push_back(h->GetBinContent(i));
      pname.push_back(CodetoName(i-3001));
    }
  }
  int sum = 0;
  for (int i = 0; i<ptcl.size(); i++){
    sum += h->GetBinContent(axis->FindBin(ptcl[i]));
    cout << ptcl[i] << " " << pname[i] << " " << nptcl[i] << endl;
  }
  cout << "ptcl.size() = " << ptcl.size() << endl;

  //TPie *pie = new TPie("pie", "Tertiary Beam Component", ptcl.size(), &nptcl[0]);
  //pie->Draw("r");
  //pie->SetAngularOffset(20.);
  //pie->SetEntryRadiusOffset( 4, 0.5);
  //pie->SetRadius(0.2);
  //pie->SetLabelsOffset(.08);
  //pie->SetLabelFormat("#splitline{%val (%perc)}{%txt}");
  //pie->Draw("nol <");
  //pie->SetEntryLineColor(2,2);
  //pie->SetEntryLineWidth(2,5);
  //pie->SetEntryLineStyle(2,2);
  //pie->SetEntryFillStyle(1,3030);
  //pie->SetCircle(.5,.45,.3);
  //pie->Draw("3d");
}

string CodetoName (int a){
  string ptcl;
  switch(a){
  case 13:
    ptcl="mu-";
    break;

  case -13:
    ptcl="mu+";
    break;

  case 11:
    ptcl="e-";
    break;

  case -11:
    ptcl="e+";
    break;

  case -211:
    ptcl="pi-";
    break;

  case 111:
    ptcl="pi0";
    break;

  case 211:
    ptcl="pi+";
    break;

  case -321:
    ptcl="K-";
    break;

  case 321:
    ptcl="K+";
    break;

  case -2212:
    ptcl="antiproton";
    break;

  case 2212:
    ptcl="proton";
    break;

  case 22:
    ptcl="gamma";
    break;

  case 2112:
    ptcl="n";
    break;

  default:
    ptcl="null";
  }
  return (ptcl);
}
