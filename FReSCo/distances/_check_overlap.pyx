"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings
from distance_enum import Distance
from distance_utils import get_ncellsx_scale

cdef class _Cdef_CheckOverlapInterface(object):
    """
    """
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_overlapping_particles(self, coords):
        cdef vector[size_t] cpp_vector = self.baseptr.get().get_overlapping_particles(coords)
        cdef np.ndarray[size_t, ndim=1] array = np.asarray(cpp_vector, dtype='uint64')
        return array

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_overlapping_particles_ca(self, coords, changed_atoms_idx):
        cdef vector[size_t] cpp_vector = self.baseptr.get().get_overlapping_particles_ca(coords, changed_atoms_idx)
        cdef np.ndarray[size_t, ndim=1] array = np.asarray(cpp_vector, dtype='uint64')
        return array

cdef class _Cdef_CheckOverlap(_Cdef_CheckOverlapInterface):

    def __cinit__(self, np.ndarray radii, np.ndarray boxv, use_cell_lists = False, ncellx_scale=None, method=Distance.PERIODIC):
        self.boxv = np.asarray(boxv, dtype='d')
        self.radii = np.asarray(radii, dtype='d')
        self.use_cell_lists = use_cell_lists
        if ncellx_scale is None:
            self.ncellx_scale = get_ncellsx_scale(radii, boxv)
            print("CheckOverlap setting ncellx_scale to value: ", self.ncellx_scale)
        else:
            self.ncellx_scale = ncellx_scale
        self.method = method
        self.ndim = self.boxv.size()
        self.natoms = self.radii.size()

        if method == Distance.CARTESIAN:
            if use_cell_lists:
                if self.ndim == 1:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesianCellLists[INT1](self.radii, self.boxv, self.ncellx_scale))
                elif self.ndim == 2:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesianCellLists[INT2](self.radii, self.boxv, self.ncellx_scale))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesianCellLists[INT3](self.radii, self.boxv, self.ncellx_scale))
                elif self.ndim == 4:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesianCellLists[INT4](self.radii, self.boxv, self.ncellx_scale))
                else:
                    raise NotImplementedError
            else:
                if self.ndim == 1:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesian[INT1](self.radii))
                elif self.ndim == 2:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesian[INT2](self.radii))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesian[INT3](self.radii))
                elif self.ndim == 4:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapCartesian[INT4](self.radii))
                else:
                    raise NotImplementedError
        elif method == Distance.PERIODIC:
            if use_cell_lists:
                if self.ndim == 1:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodicCellLists[INT1](self.radii, self.boxv, self.ncellx_scale))
                elif self.ndim == 2:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodicCellLists[INT2](self.radii, self.boxv, self.ncellx_scale))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodicCellLists[INT3](self.radii, self.boxv, self.ncellx_scale))
                elif self.ndim == 4:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodicCellLists[INT4](self.radii, self.boxv, self.ncellx_scale))
                else:
                    raise NotImplementedError
            else:
                if self.ndim == 1:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodic[INT1](self.radii, self.boxv))
                elif self.ndim == 2:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodic[INT2](self.radii, self.boxv))
                elif self.ndim == 3:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodic[INT3](self.radii, self.boxv))
                elif self.ndim == 4:
                    self.baseptr = shared_ptr[cppCheckOverlapInterface](<cppCheckOverlapInterface*> new \
                        cppCheckOverlapPeriodic[INT4](self.radii, self.boxv))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.radii, self.boxv, self.use_cell_lists, self.ncellx_scale, self.method,), d)

    def __setstate__(self, d):
        pass

class CheckOverlap(_Cdef_CheckOverlap):
    """
    python wrapper to CheckOverlap
    """
