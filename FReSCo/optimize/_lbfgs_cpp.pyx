"""
# distutils: language = C++
"""
import sys

import numpy as np

from FReSCo.potentials import _fresco
from FReSCo.potentials cimport _fresco
from FReSCo.optimize import Result
from FReSCo.potentials._pythonpotential import as_cpp_potential

cimport numpy as np
cimport FReSCo.optimize.opt as opt
from FReSCo.optimize.opt cimport shared_ptr
cimport cython
from cpython cimport bool as cbool
from libcpp.vector cimport vector

# import the externally defined ljbfgs implementation
cdef extern from "FReSCo/lbfgs.hpp" namespace "fresco":
    cdef cppclass cppLBFGS "fresco::LBFGS":
        cppLBFGS(shared_ptr[_fresco.cppBasePotential], vector[double]&, double, int) except +

        void set_H0(double) except +
        void set_tol(double) except +
        void set_maxstep(double) except +
        void set_max_f_rise(double) except +
        void set_use_relative_f(int) except +
        void set_max_iter(int) except +
        void set_iprint(int) except +
        void set_verbosity(int) except +

        double get_H0() except +



cdef class _Cdef_LBFGS_CPP(opt.GradientOptimizer):
    """This class is the python interface for the c++ LBFGS implementation
    """
    cdef _fresco.BasePotential pot
    
    def __cinit__(self, x0, potential, double tol=1e-5, int M=4, double maxstep=0.1, 
                  double maxErise=1e-4, double H0=0.1, int iprint=-1,
                  energy=None, gradient=None,
                  int nsteps=10000, int verbosity=0, events=None, logger=None,
                  rel_energy=False):
        potential = as_cpp_potential(potential, verbose=verbosity>0)

        self.pot = potential
        if logger is not None:
            print "warning c++ LBFGS is ignoring logger"
        self.thisptr = shared_ptr[opt.cGradientOptimizer]( <opt.cGradientOptimizer*>
                new cppLBFGS(self.pot.thisptr, x0,tol, M) )
        cdef cppLBFGS* lbfgs_ptr = <cppLBFGS*> self.thisptr.get()
        lbfgs_ptr.set_H0(H0)
        lbfgs_ptr.set_maxstep(maxstep)
        lbfgs_ptr.set_max_f_rise(maxErise)
        lbfgs_ptr.set_max_iter(nsteps)
        lbfgs_ptr.set_verbosity(verbosity)
        lbfgs_ptr.set_iprint(iprint)
        if rel_energy:
            lbfgs_ptr.set_use_relative_f(1)
        
        if energy is not None and gradient is not None:
            self.thisptr.get().set_func_gradient(energy, gradient)

        self.events = events
        if self.events is None: 
            self.events = []
    
    def set_H0(self, H0):
        cdef cppLBFGS* lbfgs_ptr = <cppLBFGS*> self.thisptr.get()
        lbfgs_ptr.set_H0(float(H0))
    
    def get_result(self):
        cdef cppLBFGS* lbfgs_ptr = <cppLBFGS*> self.thisptr.get()
        res = super(_Cdef_LBFGS_CPP, self).get_result()
        res["H0"] = float(lbfgs_ptr.get_H0())
        return res

class LBFGS_CPP(_Cdef_LBFGS_CPP):
    """This class is the python interface for the c++ LBFGS implementation
    """
