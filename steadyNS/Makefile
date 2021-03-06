CPP_DIR=cpp
CPP_INCLUDEDIR=$(CPP_DIR)
CPPFLAGS=-std=c++11 -I $(CPP_INCLUDEDIR) -fopenmp -fPIC -Wall -O3

PY_INCLUDEDIR=$(shell python -c \
			  "from distutils import sysconfig; \
			  print(sysconfig.get_python_inc())")
PY_LIBS=$(shell python -c \
		"from distutils import sysconfig; \
		print(sysconfig.get_config_var('LIBS'))")
PY_LDFLAGS=$(shell python -c \
		  "from distutils import sysconfig; \
		  print(sysconfig.get_config_var('LDFLAGS'))")

all:steadyNS.so mesh_extension.so poisson.so utils.so
utils.so:utils.o libutils.a
	+g++ utils.o cpp/libutils.a -o utils.so -fopenmp -shared $(PY_LIBS)
steadyNS.so:steadyNS.o libsteadyNS.a
	+g++ steadyNS.o cpp/libsteadyNS.a -o steadyNS.so -fopenmp -shared $(PY_LIBS)
mesh_extension.so:mesh_extension.o libmesh_extension.a
	+g++ mesh_extension.o cpp/libmesh_extension.a -o mesh_extension.so -fopenmp -shared $(PY_LIBS)
poisson.so:poisson.o libpoisson.a
	+g++ poisson.o cpp/libpoisson.a -o poisson.so -fopenmp -shared $(PY_LIBS)

utils.cpp:utils.pyx
	+cython utils.pyx --cplus -3
utils.o:utils.cpp
	+g++ -c utils.cpp $(CPPFLAGS) -I $(PY_INCLUDEDIR)
steadyNS.cpp:steadyNS.pyx steadyNS.pxd
	+cython steadyNS.pyx --cplus -3
steadyNS.o:steadyNS.cpp
	+g++ -c steadyNS.cpp $(CPPFLAGS) -I $(PY_INCLUDEDIR)
mesh_extension.cpp:mesh_extension.pyx mesh_extension.pxd
	+cython mesh_extension.pyx --cplus -3
mesh_extension.o:mesh_extension.cpp
	+g++ -c mesh_extension.cpp $(CPPFLAGS) -I $(PY_INCLUDEDIR)
poisson.cpp:poisson.pyx poisson.pxd
	+cython poisson.pyx --cplus -3
poisson.o:poisson.cpp
	+g++ -c poisson.cpp $(CPPFLAGS) -I $(PY_INCLUDEDIR)

libutils.a:
	+make libutils.a -C $(CPP_DIR)
libsteadyNS.a:
	+make libsteadyNS.a -C $(CPP_DIR)
libmesh_extension.a:
	+make libmesh_extension.a -C $(CPP_DIR)
libpoisson.a:
	+make libpoisson.a -C $(CPP_DIR)

.PHONY:clean
clean:
	-rm -rf steadyNS.cpp mesh_extension.cpp poisson.cpp utils.cpp 
	-rm -rf *.o steadyNS.so mesh_extension.so poisson.so utils.so __pycache__
	+make clean -C $(CPP_DIR)
