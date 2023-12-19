from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport fresco.potentials._fresco as _fresco

cdef extern from "fresco/combine_potentials.hpp" namespace "ha":
    cdef cppclass cppCombinedPotential "fresco::CombinedPotential":
        cppCombinedPotential() except +
        double get_energy(vector[double] &x) except +
        double get_energy_gradient(vector[double] &x, vector[double] &grad) except +
        void add_potential(shared_ptr[_fresco.cppBasePotential]) except +
