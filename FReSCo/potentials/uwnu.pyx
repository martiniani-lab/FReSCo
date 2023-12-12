"""
# distutils: language = C++
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class _Cdef_UwNU(_fresco.BasePotential):
    def __cinit__(self, np.ndarray N, np.ndarray K, np.ndarray Sk, np.ndarray V, 
                        np.ndarray L, double eps=1e-6):
        self.N = N
        self.K = K.ravel()
        self.Sk = Sk.ravel()
        self.V = V.ravel()
        assert len(self.Sk) == len(self.V)
        self.L = L
        self.ndim = len(L)
        self.eps = eps
        self.grad = np.zeros(N).ravel()
        self.thisptr = shared_ptr[_fresco.cppBasePotential](<_fresco.cppBasePotential*>new 
        cppUwNU(self.N, self.K, self.Sk, self.V, self.L, self.eps))

    def get_energy(self, x):
        cdef double energy = (<cppUwNU*>self.thisptr.get()).get_energy(x)
        return energy

    def get_energy_gradient(self, x):
        cdef double energy = (<cppUwNU*>self.thisptr.get()).get_energy_gradient(x, self.grad)
        return energy, self.grad

    def __reduce__(self):
        d = {}
        return (self.__class__, (self.N, self.K, self.Sk, self.V, self.L, self.eps), d)

    def __setstate__(self, d):
        pass


class UwNU(_Cdef_UwNU):
    """
    python wrapper to UwNU
    """
