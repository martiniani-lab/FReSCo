cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from ctypes import c_size_t as size_t

#===============================================================================
# shared pointer
#===============================================================================
cdef extern from "<memory>" namespace "std":
    cdef cppclass shared_ptr[T]:
        shared_ptr() except+
        shared_ptr(T*) except+
        T* get() except+
        # T& operator*() # doesn't do anything
        # Note: operator->, operator= are not supported


#===============================================================================
# fresco::BasePotential
#===============================================================================
cdef extern from "fresco/base_potential.hpp" namespace "fresco":
    cdef cppclass  cppBasePotential "fresco::BasePotential":
        cppBasePotential() except +
        double get_energy(vector[double] &x) except +
        double get_energy_gradient(vector[double] &x, vector[double] &grad) except +
        void get_neighbors(vector[double]&, vector[vector[size_t]]&, vector[vector[vector[double]]]&, double) except +
        void get_neighbors_picky(vector[double]&, vector[vector[size_t]]&, vector[vector[vector[double]]]&, vector[short]&, double) except +


#===============================================================================
# cython BasePotential
#===============================================================================
cdef class BasePotential:
    cdef shared_ptr[cppBasePotential] thisptr      # hold a C++ instance which we're wrapping

#===============================================================================
# fresco::CombinedPotential
#===============================================================================
cdef extern from "fresco/combine_potentials.hpp" namespace "fresco":
    cdef cppclass  cppCombinedPotential "fresco::CombinedPotential":
        cppCombinedPotential() except +
        double get_energy(vector[double] &x) except +
        double get_energy_gradient(vector[double] &x, vector[double] &grad) except +
        void add_potential(shared_ptr[cppBasePotential] potential) except +
