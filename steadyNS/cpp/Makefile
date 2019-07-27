SOURCES=$(wildcard *.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SOURCES))

CPPFLAGS=-std=c++11 -fPIC -O3 -Wall -fopenmp -lm -I ASTen

all:libmesh_extension.a libsteadyNS.a libpoisson.a libutils.a

$(OBJS):%.o:%.cpp
	g++ -c $(CPPFLAGS) $< -o $@
libmesh_extension.a:mesh_extension.o
	ar cr libmesh_extension.a mesh_extension.o 
libsteadyNS.a:ns-qgu.o ns-ugu.o utils.o ns-sourceF.o
	ar cr libsteadyNS.a ns-qgu.o ns-ugu.o utils.o ns-sourceF.o
libpoisson.a:poisson.o utils.o
	ar cr libpoisson.a poisson.o utils.o
libutils.a:utils.o
	ar cr libutils.a utils.o

.PHONY:clean
clean:
	-rm -rf *.o
	-rm -rf *.a
