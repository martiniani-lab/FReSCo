"""
# distutils: language = C++
"""
import numpy as np

cimport numpy as np
cimport numpy as cnp
cimport cython
from cpython.ref cimport PyObject

from fresco.potentials cimport _fresco
from fresco.potentials._fresco cimport shared_ptr

# import the numpy array api.  this is necessary because some files
# used by this module use the numpy c-api.
# see the miscillaneous section and "importing the api"
# http://docs.scipy.org/doc/numpy/reference/c-api.array.html
# also see http://mail.scipy.org/pipermail/numpy-discussion/2011-December/059612.html
# question: should I define PY_ARRAY_UNIQUE_SYMBOL?  And how can I?
cnp.import_array()

cdef extern from "fresco/python_potential_wrapper.hpp" namespace "ha":
    cdef cppclass  cppPythonPotential "fresco::PythonPotential":
        cppPythonPotential(PyObject *potential) except +

cdef class _cdef_CppPotentialWrapper(_fresco.BasePotential):
    def __cinit__(self, *args, **kwargs):
        self.thisptr = shared_ptr[_fresco.cppBasePotential]( <_fresco.cppBasePotential*>new cppPythonPotential(
                                           <PyObject*>self) )

class CppPotentialWrapper(_cdef_CppPotentialWrapper):
    """wrap a python potential to be used in c++"""
    def __init__(self, potential):
        # overload the 
        self.potential = potential
        self.getEnergy = potential.getEnergy
        self.getEnergyGradient = potential.getEnergyGradient
#        self.getEnergyGradientHessian = potential.getEnergyGradientHessian
#         self.NumericalGradient = pot.NumericalGradient


class _TestingCppPotentialWrapper(CppPotentialWrapper):
    """testing potential which provides direct access to c++ wrapper"""
    def getEnergy(self, x):
        print "going through cpp"
        return _fresco.BasePotential.getEnergy(self, x)

    def cpp_get_energy(self, x):
        return _fresco.BasePotential.getEnergy(self, x)

    def cpp_get_energy_gradient(self, x):
        return _fresco.BasePotential.getEnergyGradient(self, x)

    def getEnergyGradient(self, x):
        return _fresco.BasePotential.getEnergyGradient(self, x)
        

def as_cpp_potential(potential, verbose=False):
    """wrap a potential if necessary to it can be used with the c++ routines"""
    if issubclass(potential.__class__, _fresco.BasePotential):
        return potential
    else:
        if verbose:
            print "potential is not subclass of c++ BasePotential; wrapping it.", potential
        return CppPotentialWrapper(potential)
