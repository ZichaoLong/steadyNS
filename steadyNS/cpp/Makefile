CPPFLAGS=-fPIC -O3 -fopenmp -lm -I ASTen
all:libsteadyNS.so
preprocess.o:preprocess.cpp
	g++ -std=c++11 -c preprocess.cpp $(CPPFLAGS)
libsteadyNS.so:preprocess.o
	g++ preprocess.o -o libsteadyNS.so -shared $(CPPFLAGS)

.PHONY:clean
clean:
	-rm -rf preprocess.o libsteadyNS.so
