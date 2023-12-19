"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_UwU(_fresco.BasePotential):
    def __cinit__(self, np.ndarray N, double K, np.ndarray Sk, np.ndarray V, 
                        np.ndarray L, double eps=1e-6):
        self.K = K
        self.N = N
        self.Sk = Sk.ravel()
        self.V = V.ravel()
        assert len(self.Sk) == len(self.V)
        self.L = L
        self.ndim = len(L)
        self.eps = eps
        self.grad = np.zeros(N).ravel()
        self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*>new 
        cppUwU(self.N, self.K, self.Sk, self.V, self.L, self.eps))

    def get_energy(self, x):
        cdef double energy = (<cppUwU*>self.thisptr.get()).get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef double energy = (<cppUwU*>self.thisptr.get()).get_energy_gradient(x, self.grad)
        return energy, self.grad

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.N, self.K, self.Sk, self.V, self.L, self.eps), d)

    def __setstate__(self, d):
        pass


class UwU(_Cdef_UwU):
    """
    python wrapper to UwU
    """
