from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport FReSCo.potentials._fresco as _fresco

cdef extern from "FReSCo/nuwu.hpp" namespace "fresco":
    cdef cppclass cppNUwU "fresco::NUwU":
        cppNUwU(vector[double]&, double, vector[double]&, vector[double]&, vector[double]&, double, double, double, int, int, int, int, double, int, int) except +
        double get_energy(vector[double]&) except +
        double get_energy_gradient(vector[double]&, vector[double]&) except +

cdef class _Cdef_NUwU(_fresco.BasePotential):
    cdef public size_t ndim, N
    cdef public double K, beta, gamma, eps
    cdef int err_mode, pin_Sk, radial, periodic
    cdef vector[double] radii, Sk, V, grad, L
