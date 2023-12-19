"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings
from fresco.distances.distance_enum import Distance
from fresco.distances.distance_utils import get_ncellsx_scale
cimport fresco.potentials._fresco as _fresco
from fresco.potentials._fresco cimport shared_ptr


cdef class _Cdef_InversePower(_fresco.BasePotential):

    def __cinit__(self, double a, double eps, np.ndarray radii, np.ndarray boxv, use_cell_lists = False, ncellx_scale=None, balance_omp=True, method=Distance.PERIODIC):
        self.boxv = np.asarray(boxv, dtype='d')
        self.radii = np.asarray(radii, dtype='d')
        self.use_cell_lists = use_cell_lists
        if ncellx_scale is None:
            self.ncellx_scale = get_ncellsx_scale(radii, boxv)
            print("InversePower setting ncellx_scale to value: ", self.ncellx_scale)
        else:
            self.ncellx_scale = ncellx_scale
        self.balance_omp = balance_omp
        self.method = method
        self.ndim = self.boxv.size
        self.natoms = self.radii.size
        self.a = a
        self.eps = eps

        if method == Distance.CARTESIAN:
            if self.ndim == 1:
                self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                    cppInversePowerCartesian[INT1](self.a, self.eps, self.radii))
            elif self.ndim == 2:
                self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                    cppInversePowerCartesian[INT2](self.a, self.eps, self.radii))
            elif self.ndim == 3:
                self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                    cppInversePowerCartesian[INT3](self.a, self.eps, self.radii))
            elif self.ndim == 4:
                self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                    cppInversePowerCartesian[INT4](self.a, self.eps, self.radii))
            else:
                raise NotImplementedError
        elif method == Distance.PERIODIC:
            if use_cell_lists:
                if self.ndim == 1:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodicCellLists[INT1](self.a, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 2:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodicCellLists[INT2](self.a, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 3:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodicCellLists[INT3](self.a, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                elif self.ndim == 4:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodicCellLists[INT4](self.a, self.eps, self.radii, self.boxv, self.ncellx_scale, self.balance_omp))
                else:
                    raise NotImplementedError
            else:
                if self.ndim == 1:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodic[INT1](self.a, self.eps, self.radii, self.boxv))
                elif self.ndim == 2:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodic[INT2](self.a, self.eps, self.radii, self.boxv))
                elif self.ndim == 3:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodic[INT3](self.a, self.eps, self.radii, self.boxv))
                elif self.ndim == 4:
                    self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*> new \
                        cppInversePowerPeriodic[INT4](self.a, self.eps, self.radii, self.boxv))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.a, self.eps, self.radii, self.boxv, self.use_cell_lists, self.ncellx_scale, self.balance_omp, self.method,), d)

    def __setstate__(self, d):
        pass

class InversePower(_Cdef_InversePower):
    """
    python wrapper to InversePower
    """
