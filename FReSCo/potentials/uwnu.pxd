from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport FReSCo.potentials._fresco as _fresco

cdef extern from "FReSCo/uwnu.hpp" namespace "fresco":
    cdef cppclass cppUwNU "fresco::UwNU":
        cppUwNU(vector[int]&, vector[double]&, vector[double]&, vector[double]&, vector[double]&, double) except +
        double get_energy(vector[double]&) except +
        double get_energy_gradient(vector[double]&, vector[double]&) except +

cdef class _Cdef_UwNU(_fresco.BasePotential):
    cdef public size_t ndim
    cdef public double eps
    cdef vector[double] K, Sk, V, grad, L
    cdef vector[int] N
