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

all:steadyNS.so
steadyNS.cpp:steadyNS.pyx
	cython steadyNS.pyx --cplus -3
steadyNS.o:steadyNS.cpp
	g++ -c steadyNS.cpp -O3 -fPIC -I $(CPP_INCLUDEDIR) -I $(PY_INCLUDEDIR)
cppextension:
	+make -C $(CPP_DIR)
steadyNS.so:steadyNS.o cppextension
	g++ *.o cpp/*.o -o steadyNS.so -fopenmp -shared $(PY_LIBS)

.PHONY:clean
clean:
	-rm -rf steadyNS.cpp *.o steadyNS.so __pycache__
	+make clean -C $(CPP_DIR)
