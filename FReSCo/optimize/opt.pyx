# distutils: language = c++
import sys

import numpy as np
import hickle as hkl
#from FReSCo.potentials._fresco import *
from FReSCo.optimize import Result

cimport numpy as np
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
cimport cython
import copy

#from FReSCo.potentials cimport _fresco

cdef class GradientOptimizer(object):
    """this class defines the python interface for c++ gradient optimizers 
    
    Notes
    -----
    for direct access to the underlying c++ optimizer use self.thisptr
    """
    res = None

    def one_iteration(self):
        self.thisptr.get().one_iteration()
        self.res = self.get_result()
        for event in self.events:
            event(coords=self.res.coords, energy=self.res.energy, rms=self.res.rms)
        return self.res
        
    def run(self, niter=None, isave=None, file_name = 'unknown'):
        if not self.events:
            # if we don't need to call python events then we can
            # go just let the c++ optimizer do it's thing
            if niter is None and isave is None:
                self.thisptr.get().run()
            elif isave is None:
                self.thisptr.get().run(niter)
            else:
                if niter is None:
                    niter = self.get_maxiter() - self.get_niter()
                i = 0
                odd = True
                while i < niter:
                    self.thisptr.get().run(isave)
                    self.res = self.get_result()
                    if odd:
                        hkl.dump(self.res.coords,file_name+'_odd.hkl',mode='w')
                    else:
                        hkl.dump(self.res.coords,file_name+'_even.hkl',mode='w')
                    i += isave
                    odd = not odd
                    
            self.res = self.get_result()
        else:
            # we need to call python events after each iteration.
            if niter is None:
                niter = self.get_maxiter() - self.get_niter()
    
            for i in xrange(niter):
                if self.stop_criterion_satisfied():
                    break
                self.res = self.one_iteration()

        return copy.deepcopy(self.res)
            
    def reset(self, coords):
        self.thisptr.get().reset(coords)
    
    def stop_criterion_satisfied(self):
        return bool(self.thisptr.get().stop_criterion_satisfied())

    def get_maxiter(self):
        return self.thisptr.get().get_maxiter()

    def get_niter(self):
        return self.thisptr.get().get_niter()
    
    def get_result(self):
        """return a results object"""
        res = Result()
        
        res.energy = self.thisptr.get().get_f()
        res.coords = self.thisptr.get().get_x()
        res.grad   = self.thisptr.get().get_g()
        
        res.rms = self.thisptr.get().get_rms()
        res.nsteps = self.thisptr.get().get_niter()
        res.nfev = self.thisptr.get().get_nfev()
        res.success = bool(self.thisptr.get().success())
        return res
