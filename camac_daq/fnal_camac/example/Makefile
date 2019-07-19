
all	: dma pdt qxtab cnaf bypass cz 2249a

CFLAGS = -I../include -g -Wall -ansi

FFLAGS = -Wf,-I../include -g

.c.o:
	$(CC) $(CFLAGS) -c $*.c

LIBRARY = ../lib/libsjy.a

2249a		: 2249a.o $(LIBRARY)
	$(CC) $(CFLAGS) 2249a.o $(LIBRARY) -o $@

io		: io.o $(LIBRARY)
	$(CC) $(CFLAGS) io.o $(LIBRARY) -o $@

dma		: dma.o $(LIBRARY)
	$(CC) $(CFLAGS) dma.o $(LIBRARY) -o $@

pdt		: pdt.o $(LIBRARY)
	$(CC) $(CFLAGS) pdt.o $(LIBRARY) -o $@

qxtab: qxtab.o $(LIBRARY)
	$(CC) $(CFLAGS) qxtab.o $(LIBRARY) -o $@

cnaf: cnaf.o $(LIBRARY)
	$(CC) $(CFLAGS) cnaf.o $(LIBRARY) -o $@

bypass: bypass.o $(LIBRARY)
	$(CC) $(CFLAGS) bypass.o $(LIBRARY) -o $@

cz: cz.o $(LIBRARY)
	$(CC) $(CFLAGS) cz.o $(LIBRARY) -o $@

