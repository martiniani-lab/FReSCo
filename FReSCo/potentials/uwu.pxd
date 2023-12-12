from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport FReSCo.potentials._fresco as _fresco

cdef extern from "FReSCo/uwu.hpp" namespace "fresco":
    cdef cppclass cppUwU "fresco::UwU":
        cppUwU(vector[int]&, double, vector[double]&, vector[double]&, vector[double]&, double, double, double, int, int, int, int, double) except +
        double get_energy(vector[double]&) except +
        double get_energy_gradient(vector[double]&, vector[double]&) except +

cdef class _Cdef_UwU(_fresco.BasePotential):
    cdef public size_t ndim
    cdef public double K, beta, gamma, eps
    cdef int err_mode, pin_Sk
    cdef vector[double] Sk, V, grad, L
    cdef vector[int] N
