#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;
string CodetoName (int a);

void momentum() {
  gStyle->SetOptFit(0000);
  gStyle->SetStatFontSize(0.05);
  gStyle->SetOptStat("rme");
  gStyle->SetStatW(0.3);
  TGaxis::SetMaxDigits(3);
  gStyle->SetTitleFontSize(0.05);
  setH2ColorStyle();

  string figureDir = "/Users/juntinghuang/google_drive/slides/beamer/20170713_g4beamline_nova_nd/figures";
  string dir = "/Users/juntinghuang/Desktop/nova/testbeam/";
  string addr = "data/beam.1_spill.root";
  rootname = dir + addr;
  cout<<"reading "<<rootname<<endl;
  TFile* f = new TFile(rootname.c_str());
  f->cd("VirtualDetector");
  TTree* nt = (TTree*) gDirectory->Get("TOFds");

  TH1I* h = new TH1I("h", "h", 6000, -3000, 3000);
  nt->Draw("PDGid >> h");
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
  int sum;
  for (int i = 0; i<ptcl.size(); i++){
    sum += h->GetBinContent(axis->FindBin(ptcl[i]));
    // cout << setw(10) << ptcl[i] << setw(10) << pname[i] << setw(10) << nptcl[i] << endl;
    cout << setw(10) << ptcl[i] << "&" << setw(10) << pname[i] << "&" << setw(10) << nptcl[i] << "\\\\" << endl;
  }
  cout << "ptcl.size() = " << ptcl.size() << endl;
  cout << "nt->GetEntries() = " << nt->GetEntries() << " sum = " << sum << endl;

  //momentum info
  //input
  // int pcode=ptcl[8];
  // int pcode=ptcl[4];
  // int pcode=11;
  // int pcode=-211;
  // int pcode=211;
  int pcode=2212;
  //end of input

  double titleOffset = 2.5;

  double z0 = 8005.9022;
  double x0 = -1186.1546;

  char psel[50];
  sprintf(psel,"PDGid == %d", pcode);
  TCanvas *c=new TCanvas(CodetoName(pcode).c_str(), CodetoName(pcode).c_str(), 1600, 1000);
  c->Divide(2,2);
  c->cd(1);
  setMargin();
  TH2F* XY = new TH2F("XY", "XY",
                      100, x0 - 200, x0 + 200,
                      100, -200, 200);
  nt->Draw("y:x >> XY", psel);
  XY->Draw("colz");
  XY->SetTitle(CodetoName(pcode).c_str());
  XY->GetXaxis()->SetTitle("X (mm)");
  XY->GetYaxis()->SetTitle("Y (mm)");
  setH2Style(XY);
  XY->GetXaxis()->SetTitleOffset(titleOffset);
  XY->GetYaxis()->SetTitleOffset(titleOffset);
  c->Update();
  TPaveStats *p1 = (TPaveStats*)XY->GetListOfFunctions()->FindObject("stats");
  XY->GetListOfFunctions()->Remove(p1);
  XY->SetStats(0);
  p1->SetX1NDC(0.65);
  p1->SetX2NDC(0.88);
  p1->SetY1NDC(0.58);
  p1->SetY2NDC(0.88);
  p1->Draw();

  c->cd(2);
  setMargin();
  TH2F* Pxy = new TH2F("Pxy", "Pxy", 120, -120, 120, 120, -120, 120);
  nt->Draw("Py:Px >> Pxy",psel);
  Pxy->Draw("colz");
  Pxy->SetTitle(CodetoName(pcode).c_str());
  Pxy->GetXaxis()->SetTitle("P_{x} (MeV/c)");
  Pxy->GetYaxis()->SetTitle("P_{y} (MeV/c)");
  setH2Style(Pxy);
  Pxy->GetXaxis()->SetTitleOffset(titleOffset);
  Pxy->GetYaxis()->SetTitleOffset(titleOffset);
  c->Update();
  TPaveStats *p2 = (TPaveStats*)Pxy->GetListOfFunctions()->FindObject("stats");
  Pxy->GetListOfFunctions()->Remove(p2);
  Pxy->SetStats(0);
  p2->SetX1NDC(0.65);
  p2->SetX2NDC(0.88);
  p2->SetY1NDC(0.58);
  p2->SetY2NDC(0.88);
  p2->Draw();

  c->cd(3);
  setMargin();
  TH2F* Pzt = new TH2F("Pzt", "Pzt", 120, 0, 120, 120, 0, 1200);
  nt->Draw("Pz:pow(pow(Px,2)+pow(Py,2),0.5) >> Pzt", psel);
  Pzt->Draw("colz");
  Pzt->SetTitle(CodetoName(pcode).c_str());
  Pzt->GetXaxis()->SetLabelSize(0.05);
  Pzt->GetXaxis()->SetTitleSize(0.05);
  Pzt->GetYaxis()->SetLabelSize(0.05);
  Pzt->GetYaxis()->SetTitleSize(0.05);
  Pzt->GetXaxis()->SetTitle("P_{t} (MeV/c)");
  Pzt->GetYaxis()->SetTitle("P_{z} (MeV/c)");
  setH2Style(Pzt);
  Pzt->GetXaxis()->SetTitleOffset(titleOffset);
  Pzt->GetYaxis()->SetTitleOffset(titleOffset);
  c->Update();
  TPaveStats *p3 = (TPaveStats*)Pzt->GetListOfFunctions()->FindObject("stats");
  Pzt->GetListOfFunctions()->Remove(p3);
  Pzt->SetStats(0);
  p3->SetX1NDC(0.65);
  p3->SetX2NDC(0.88);
  p3->SetY1NDC(0.58);
  p3->SetY2NDC(0.88);
  p3->Draw();

  c->cd(4);
  setMargin();
  gPad->SetLogy();
  TH1F* P = new TH1F("P", "P", 130, -100, 1200);
  nt->Draw("pow(pow(Px, 2) + pow(Py, 2) + pow(Pz, 2), 0.5) >> P", psel);
  P->Draw();
  P->SetTitle(CodetoName(pcode).c_str());
  P->GetXaxis()->SetLabelSize(0.05);
  P->GetXaxis()->SetTitleSize(0.05);
  P->GetYaxis()->SetLabelSize(0.05);
  P->GetYaxis()->SetTitleSize(0.05);
  P->GetXaxis()->SetTitle("P (MeV/c)");
  P->GetYaxis()->SetTitle("Particle Count");
  setH1Style(P);
  P->GetXaxis()->SetTitleOffset(titleOffset);
  P->GetYaxis()->SetTitleOffset(titleOffset);
  c->Update();
  TPaveStats *p4 = (TPaveStats*)P->GetListOfFunctions()->FindObject("stats");
  P->GetListOfFunctions()->Remove(p4);
  P ->SetStats(0);
  p4->SetX1NDC(0.65);
  p4->SetX2NDC(0.88);
  p4->SetY1NDC(0.70);
  p4->SetY2NDC(0.88);
  p4->Draw();

  string eps = figureDir + "/" + CodetoName(pcode) + ".pdf";
  c->SaveAs(eps.c_str());
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
