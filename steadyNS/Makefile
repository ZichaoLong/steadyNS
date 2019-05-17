CPP_DIR=../steadyNS/cpp
# CPP_DIR=$(shell pwd)/cpp
CPP_INCLUDEDIR=$(CPP_DIR)
CPP_LIB=-lsteadyNS
CPP_LDFLAGS=-L $(CPP_DIR) -Wl,-rpath=$(CPP_DIR) -Wl,--no-as-needed

PY_INCLUDEDIR=$(shell python -c \
			  "from distutils import sysconfig; \
			  print(sysconfig.get_python_inc())")
PY_LIBS=$(shell python -c \
		"from distutils import sysconfig; \
		print(sysconfig.get_config_var('LIBS'))")
PY_LDFLAGS=$(shell python -c \
		  "from distutils import sysconfig; \
		  print(sysconfig.get_config_var('LDFLAGS'))")

all:steadyNS.so
steadyNS.cpp:steadyNS.pyx
	cython steadyNS.pyx --cplus 
steadyNS.o:steadyNS.cpp
	g++ -c steadyNS.cpp -O3 -fPIC -I $(CPP_INCLUDEDIR) -I $(PY_INCLUDEDIR)
libsteadyNS.so:
	+make -C $(CPP_DIR)
steadyNS.so:steadyNS.o libsteadyNS.so
	g++ steadyNS.o -o steadyNS.so -fopenmp -shared $(CPP_LIB) $(PY_LIBS) $(CPP_LDFLAGS) $(PY_LDFLAGS)

.PHONY:clean
clean:
	-rm -rf steadyNS.cpp steadyNS.o steadyNS.so __pycache__
	make clean -C $(CPP_DIR)
