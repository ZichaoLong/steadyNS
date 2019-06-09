CPP_DIR=cpp
CPP_INCLUDEDIR=$(CPP_DIR)

PY_INCLUDEDIR=$(shell python -c \
			  "from distutils import sysconfig; \
			  print(sysconfig.get_python_inc())")
PY_LIBS=$(shell python -c \
		"from distutils import sysconfig; \
		print(sysconfig.get_config_var('LIBS'))")
PY_LDFLAGS=$(shell python -c \
		  "from distutils import sysconfig; \
		  print(sysconfig.get_config_var('LDFLAGS'))")

all:steadyNS.so mesh_extension.so
steadyNS.so:steadyNS.o libsteadyNS.a
	g++ steadyNS.o cpp/libsteadyNS.a -o steadyNS.so -fopenmp -shared $(PY_LIBS)
mesh_extension.so:mesh_extension.o libmesh_extension.a
	g++ mesh_extension.o cpp/libmesh_extension.a -o mesh_extension.so -fopenmp -shared $(PY_LIBS)

steadyNS.cpp:steadyNS.pyx steadyNS.pxd
	cython steadyNS.pyx --cplus -3
steadyNS.o:steadyNS.cpp
	g++ -c steadyNS.cpp -O3 -fPIC -Wall -I $(CPP_INCLUDEDIR) -I $(PY_INCLUDEDIR)
mesh_extension.cpp:mesh_extension.pyx mesh_extension.pxd
	cython mesh_extension.pyx --cplus -3
mesh_extension.o:mesh_extension.cpp
	g++ -c mesh_extension.cpp -O3 -fPIC -Wall -I $(CPP_INCLUDEDIR) -I $(PY_INCLUDEDIR)

libsteadyNS.a:
	+make libsteadyNS.a -C $(CPP_DIR)
libmesh_extension.a:
	+make libmesh_extension.a -C $(CPP_DIR)

.PHONY:clean
clean:
	-rm -rf steadyNS.cpp mesh_extension.cpp *.o steadyNS.so mesh_extension.so __pycache__
	+make clean -C $(CPP_DIR)
