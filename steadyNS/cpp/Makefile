CPPFLAGS=-std=c++11 -fPIC -O3 -Wall -fopenmp -lm -I ASTen
all:preprocess.o stiffmat.o poisson.o
preprocess.o:preprocess.cpp
	g++ -c $(CPPFLAGS) preprocess.cpp
stiffmat.o:stiffmat.cpp
	g++ -c $(CPPFLAGS) stiffmat.cpp
poisson.o:poisson.cpp
	g++ -c $(CPPFLAGS) poisson.cpp

.PHONY:clean
clean:
	-rm -rf *.o
