from libcpp cimport bool as cbool
cimport numpy as np
cimport fresco.optimize.opt as opt
from fresco.potentials cimport _fresco
from fresco.potentials._fresco cimport shared_ptr
from libcpp.vector cimport vector

# import the externally defined modified_fire implementation
cdef extern from "fresco/modified_fire.hpp" namespace "fresco":
    cdef cppclass cppMODIFIED_FIRE "fresco::MODIFIED_FIRE":
        cppMODIFIED_FIRE(shared_ptr[_fresco.cppBasePotential] , vector[double]&, 
                         double, double, double, size_t , double, double, 
                         double, double, double, cbool) except +
