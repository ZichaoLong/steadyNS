SOURCES=$(wildcard *.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SOURCES))

CPPFLAGS=-std=c++11 -fPIC -O3 -Wall -fopenmp -lm -I ASTen

all:libmesh_extension.a libsteadyNS.a

$(OBJS):%.o:%.cpp
	g++ -c $(CPPFLAGS) $< -o $@
libmesh_extension.a:mesh_extension.o
	ar cr libmesh_extension.a mesh_extension.o 
libsteadyNS.a:ns.o poisson.o
	ar cr libsteadyNS.a ns.o poisson.o

.PHONY:clean
clean:
	-rm -rf *.o
	-rm -rf *.a
