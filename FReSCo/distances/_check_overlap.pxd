from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT1 "1"    # a fake type
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type
    ctypedef int INT4 "4"    # a fake type


# use external c++ classes
cdef extern from "FReSCo/check_overlap.hpp" namespace "fresco":
    cdef cppclass cppCheckOverlapInterface "fresco::CheckOverlapInterface":
        vector[size_t] get_overlapping_particles_ca(vector[double]&, vector[long]&) except +
        vector[size_t] get_overlapping_particles(vector[double]&) except +
    cdef cppclass cppCheckOverlapCartesian "fresco::CheckOverlapCartesian"[ndim]:
        cppCheckOverlapCartesian(vector[double]&) except +
    cdef cppclass cppCheckOverlapPeriodic "fresco::CheckOverlapPeriodic"[ndim]:
        cppCheckOverlapPeriodic(vector[double]&, vector[double]&) except +
cdef extern from "FReSCo/check_overlap_cell_lists.hpp" namespace "fresco":
    cdef cppclass cppCheckOverlapCartesianCellLists "fresco::CheckOverlapCartesianCellLists"[ndim]:
        cppCheckOverlapCartesianCellLists(vector[double]&, vector[double]&, double) except +
    cdef cppclass cppCheckOverlapPeriodicCellLists "fresco::CheckOverlapPeriodicCellLists"[ndim]:
        cppCheckOverlapPeriodicCellLists(vector[double]&, vector[double]&, double) except +

cdef class _Cdef_CheckOverlapInterface:
    cdef shared_ptr[cppCheckOverlapInterface] baseptr
    cdef public size_t natoms, ndim
    cdef vector[double] boxv, radii
    cdef double ncellx_scale

# https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
