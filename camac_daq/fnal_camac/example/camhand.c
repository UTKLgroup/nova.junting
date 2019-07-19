/******** camhand.c   --  hand-made camac 
by: agatam
data: 22-sep-94  ************************/


#include "vsd_config.h"
#include <string.h>
#include <curses.h>
/****   routines must be compiled with the -lcurses option of cc.        ******/

int key, k, resp, Q, X;
long int return_status, route, ext, b=1, c=1, n=1, a=0, f=0, d=0; 

int getkey()  { int key;  while ((key = getch()) == ERR); return (key);  }
 /********************/

void outDec(int a) { if((a/10) > 0) outDec(a/10);  addch(a%10+'0');  }
 /********************/

char Hex[] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
void outHex(int a) { if((a/16) > 0) outHex(a/16);  addch(Hex[a%16]);  }
 /********************/

int readDecimal()
{/********************/ 
   int number = 0;    addch(key + 'A' - 'a');
   while ((key = getkey()) != ' ') {
     if((key >= '0') && (key <= '9')) { 
                           addch(key);   number = 10*number + key - '0';      };
   };  addch(key);  return (number);
}

int readHexadecimal()
{/********************/
   int number = 0;    addch('H');
   while ((key = getkey()) != ' ') {
     if((key >= '0') && (key <= '9')) {
                           addch(key);   number = 16*number + key - '0';      };
     if((key >= 'a') && (key <= 'f')) {
                           addch(key);   number = 16*number + key - 'a' + 10; };
   };  addch(key);  return (number);
}

main() 
{/********************/
   WINDOW *window_pointer;
   char helpstring[] = 
                "camhand :: Cxx Z Nxx Axx Fxx Dxxxxx Hxxxxxx Q <  >for run\n\r";
   char retur[] = "\n\r"; 
   int Qtb[] = { 1, 0, 1, 0 }, Xtb[] = { 1, 1, 0, 0 };
  unsigned short aaCSR;

  if ((return_status = cdchn(b, 1, &route)) !=  VSD_S_SUCCESS) {
     if (return_status == VSD_S_VME_OPEN_FAIL)
           printf (" Error in open vme return status = : %d ",return_status);
     else if (return_status == VSD_S_MMAP_FAIL)
           printf (" Error in mmap return status = : %d ",return_status);
  } else {
                     
   window_pointer = initscr();  cbreak();  noecho();  addstr(helpstring);
   do {
        switch(key = getkey()) {
          case  10 : addstr(retur);          break;
          case 'c' : c = readDecimal();      break; 
          case 'n' : n = readDecimal();      break;
          case 'a' : a = readDecimal();      break;
          case 'f' : f = readDecimal();      break;
          case 'd' : d = readDecimal();      break;
          case 'h' : d = readHexadecimal();  break;

          case 'z' : cdreg(&ext, b,c,0,0);  cccbyp(ext,0);  cccoff(ext,0);
                     ccci(ext,0);  cccc(ext);  cccz(ext);  ccci(ext,0); 
                     addstr("\n\r ZZ  "); break;

          case ' ' : addstr(retur);   cdreg(&ext, b, c, n, a);

                     if (cfsa(f, ext, &d, &resp) == VSD_S_NO_RESPONSE) {
                         addstr("\n\r NO_RESPONSE"); };
                     vsd_csr_rd( 4, &aaCSR);   Q = ((aaCSR >> 6) & 1);
                                               X = ((aaCSR >> 5) & 1);

/*                     Q = qrespn();  X = xrespn();
                     ctstat( &k );  Q = Qtb[k & 3];  X = Xtb[k & 3];      */
                     addstr(retur);                addstr(" b"); outDec(b);
                     addstr(" c"); outDec(c);      addstr(" n"); outDec(n); 
                     addstr(" a"); outDec(a);      addstr(" f"); outDec(f);
                     if((f < 8) || ((f > 15) && (f < 24))) {
                                                   addstr(" d"); outDec(d); 
                                                   addstr("=="); outHex(d);  };
                     addstr(" Q"); outDec(Q);      addstr(" X"); outDec(X);
                     addstr(" CSR"); outHex(aaCSR);    
                     addstr("  ");
        }; 
   }  while (key != 'q');   endwin();
 };  
}
