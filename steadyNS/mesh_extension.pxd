# distutils: language = c++
# cython: language_level=3

cdef extern from "steadyNS.h":
    int _switchEdgeNode(const int L, int *Edge)
    int _updateEdgeTags(const int d, const int N, const int NE, const int *Edge, 
            const int *B, const double *coord, int *Bedge)

