# distutils: language = c++

cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_PointPatternStatisticsInterface(object):
    """
    """

    def get_seed(self):
        return self.baseptr.get().get_seed()

    def set_generator_seed(self, size_t seed):
        self.baseptr.get().set_generator_seed(seed)

    def run(self, size_t n):
        self.baseptr.get().run(n)

    def get_count(self):
        return self.baseptr.get().get_count()

    def get_mean(self):
        return self.baseptr.get().get_mean()

    def get_variance(self):
        return self.baseptr.get().get_variance()

    @property
    def mean(self):
        return self.get_mean()

    @property
    def var(self):
        return self.get_variance()

    @property
    def mean_var(self):
        return self.get_mean(), self.get_variance()

    @property
    def count(self):
        return self.get_count()

cdef class _Cdef_PointPatternStatistics(_Cdef_PointPatternStatisticsInterface):
    def __cinit__(self, coords, boxv, radius, rseed):
        self.coords = np.array(coords, dtype='d')
        self.boxv = np.array(boxv, dtype='d')
        self.radius = radius
        self.ndim = self.boxv.size
        self.natoms = len(coords) // self.ndim
        self.rseed = rseed
        if self.ndim == 1:
            self.baseptr = shared_ptr[cppPointPatternStatisticsInterface](<cppPointPatternStatisticsInterface*> new \
                cppPointPatternStatistics[INT1](self.coords, self.boxv, self.radius, self.rseed))
        elif self.ndim == 2:
            self.baseptr = shared_ptr[cppPointPatternStatisticsInterface](<cppPointPatternStatisticsInterface*> new \
                cppPointPatternStatistics[INT2](self.coords, self.boxv, self.radius, self.rseed))
        elif self.ndim == 3:
            self.baseptr = shared_ptr[cppPointPatternStatisticsInterface](<cppPointPatternStatisticsInterface*> new \
                cppPointPatternStatistics[INT3](self.coords, self.boxv, self.radius, self.rseed))
        elif self.ndim == 4:
            self.baseptr = shared_ptr[cppPointPatternStatisticsInterface](<cppPointPatternStatisticsInterface*> new \
                cppPointPatternStatistics[INT4](self.coords, self.boxv, self.radius, self.rseed))
        else:
            raise NotImplementedError

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.coords, self.boxv, self.radius, self.rseed,), d)

    def __setstate__(self, d):
        pass

class PointPatternStatistics(_Cdef_PointPatternStatistics):
    """
    python wrapper to cppRandOrg
    """


cdef class _Cdef_BatchPPSInterface(object):
    """
    """

    def get_seed(self):
        return self.baseptr.get().get_seed()

    def set_generator_seed(self, size_t seed):
        self.baseptr.get().set_generator_seed(seed)

    def run(self, size_t n):
        self.baseptr.get().run(n)

    def get_count(self):
        return self.baseptr.get().get_count()

    def get_mean(self):
        return self.baseptr.get().get_mean()

    def get_variance(self):
        return self.baseptr.get().get_variance()

    @property
    def mean(self):
        return self.get_mean()

    @property
    def var(self):
        return self.get_variance()

    @property
    def mean_var(self):
        return self.get_mean(), self.get_variance()

    @property
    def count(self):
        return self.get_count()

cdef class _Cdef_BatchPPS(_Cdef_BatchPPSInterface):
    def __cinit__(self, coords, boxv, radii, rseed):
        self.coords = np.array(coords, dtype='d')
        self.boxv = np.array(boxv, dtype='d')
        self.radii = np.array(radii, dtype = 'd')
        self.ndim = self.boxv.size
        self.natoms = len(coords) // self.ndim
        self.rseed = rseed
        if self.ndim == 1:
            self.baseptr = shared_ptr[cppBatchPPSInterface](<cppBatchPPSInterface*> new \
                cppBatchPPS[INT1](self.coords, self.boxv, self.radii, self.rseed))
        elif self.ndim == 2:
            self.baseptr = shared_ptr[cppBatchPPSInterface](<cppBatchPPSInterface*> new \
                cppBatchPPS[INT2](self.coords, self.boxv, self.radii, self.rseed))
        elif self.ndim == 3:
            self.baseptr = shared_ptr[cppBatchPPSInterface](<cppBatchPPSInterface*> new \
                cppBatchPPS[INT3](self.coords, self.boxv, self.radii, self.rseed))
        elif self.ndim == 4:
            self.baseptr = shared_ptr[cppBatchPPSInterface](<cppBatchPPSInterface*> new \
                cppBatchPPS[INT4](self.coords, self.boxv, self.radii, self.rseed))
        else:
            raise NotImplementedError

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.coords, self.boxv, self.radii, self.rseed,), d)

    def __setstate__(self, d):
        pass

class BatchPPS(_Cdef_BatchPPS):
    """
    python wrapper to cppRandOrg
    """
