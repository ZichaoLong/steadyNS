CPPFLAGS=-std=c++11 -fPIC -O3 -fopenmp -lm -I ASTen
all:libsteadyNS.so
preprocess.o:preprocess.cpp
	g++ -c $(CPPFLAGS) preprocess.cpp
stiffmat.o:stiffmat.cpp
	g++ -c $(CPPFLAGS) stiffmat.cpp
poisson.o:poisson.cpp
	g++ -c $(CPPFLAGS) poisson.cpp
libsteadyNS.so:preprocess.o stiffmat.o poisson.o
	g++ preprocess.o stiffmat.o poisson.o -o libsteadyNS.so -shared $(CPPFLAGS)

.PHONY:clean
clean:
	-rm -rf preprocess.o stiffmat.o poisson.o libsteadyNS.so
