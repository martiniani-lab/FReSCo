"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings
from FReSCo.distances.distance_enum import Distance
from FReSCo.distances.distance_utils import get_ncellsx_scale
cimport FReSCo.potentials._fresco as _fresco


cdef class _Cdef_WCAPotential(_fresco.BasePotential):

    def __cinit__(self, double sigma, double eps, np.ndarray radii, np.ndarray boxv, use_cell_lists = False, ncellx_scale=None, balance_omp=True, method=Distance.PERIODIC):
        self.boxv = np.asarray(boxv, dtype='d')
        self.radii = np.asarray(radii, dtype='d')
        self.use_cell_lists = use_cell_lists
        if ncellx_scale is None:
            self.ncellx_scale = get_ncellsx_scale(radii, boxv)
            print("WCA setting ncellx_scale to value: ", self.ncellx_scale)
        else:
            self.ncellx_scale = ncellx_scale
        self.balance_omp = balance_omp
        self.method = method
        self.natoms = self.radii.size()
        self.ndim = self.boxv.size()
        self.sigma = sigma
        self.eps = eps

        if method == Distance.CARTESIAN:
            if self.ndim == 2:
                self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                    cppWCA[INT2](self.sigma, self.eps, self.radii))
            elif self.ndim == 3:
                self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                    cppWCA[INT3](self.sigma, self.eps, self.radii))
            else:
                raise NotImplementedError
        elif method == Distance.PERIODIC:
            if use_cell_lists:
                if self.ndim == 2:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppWCAPeriodicCellLists[INT2](self.sigma, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 3:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppWCAPeriodicCellLists[INT3](self.sigma, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                else:
                    raise NotImplementedError
            else:
                if self.ndim == 2:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppWCAPeriodic[INT2](self.sigma, self.eps, self.radii, self.boxv))
                elif self.ndim == 3:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppWCAPeriodic[INT3](self.sigma, self.eps, self.radii, self.boxv))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
    
    def __reduce__(self):
        d = {}
        return (self.__class__, (self.sigma, self.eps, self.radii, self.boxv, self.use_cell_lists, self.ncellx_scale, self.balance_omp, self.method,), d)

    def __setstate__(self, d):
        pass

class WCAPotential(_Cdef_WCAPotential):
    """
    python wrapper to WCA
    """
