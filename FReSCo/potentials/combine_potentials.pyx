
"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings    
        
cdef class _Cdef_CombinedPotential(_fresco.BasePotential):
    def __cinit__(self):
        self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*>new 
        cppCombinedPotential())
        
    def add_potential(self, _fresco.BasePotential potential):
        (<cppCombinedPotential*>self.thisptr.get()).add_potential(potential.thisptr)
    
    def get_energy(self, x):
        cdef double energy = (<cppCombinedPotential*>self.thisptr.get()).get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef vector[double] grad = np.zeros(x.size)
        cdef double energy = (<cppCombinedPotential*>self.thisptr.get()).get_energy_gradient(x, grad)
        return energy, grad

class CombinedPotential(_Cdef_CombinedPotential):
    """
    Python wrapper for CombinedPotential
    """
