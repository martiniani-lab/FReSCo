from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np

cimport FReSCo.potentials._fresco as _fresco
from FReSCo.potentials._fresco cimport shared_ptr

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

cdef extern from "FReSCo/inversepower_potential.hpp" namespace "fresco":
    cdef cppclass cppInversePowerCartesian "fresco::InversePowerCartesian"[ndim]:
        cppInversePowerCartesian(double, double, vector[double]&) except +
    cdef cppclass cppInversePowerPeriodic "fresco::InversePowerPeriodic"[ndim]:
        cppInversePowerPeriodic(double, double, vector[double]&, vector[double]&) except +
    cdef cppclass cppInversePowerPeriodicCellLists "fresco::InversePowerPeriodicCellLists"[ndim]:
        cppInversePowerPeriodicCellLists(double, double, vector[double]&, vector[double]&, double, bint) except +

cdef class _Cdef_InversePowerPotential(_fresco.BasePotential):
    cdef public size_t natoms, ndim
    cdef vector[double] boxv, radii
    cdef double a, eps, ncellx_scale
    cdef bint balance_omp

# https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
