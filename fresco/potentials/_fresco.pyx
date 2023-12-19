"""
# distutils: language = C++
basic potential interface stuff
"""
cimport cython
cimport numpy as np
import numpy as np
import warnings

cdef class BasePotential(object):
    """
    """
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_energy(self, x):
        cdef double energy = self.thisptr.get().get_energy(x)
        return energy

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_energy_gradient(self, x):
        cdef vector[double] grad = np.zeros(self.natoms*self.ndim)
        cdef double energy = self.thisptr.get().get_energy_gradient(x, grad)
        return energy, grad

    def getNeighbors(self, np.ndarray[double, ndim=1] coords not None,
                      include_atoms=None, cutoff_factor=1.0):
        cdef vector[vector[size_t]] c_neighbor_indss
        cdef vector[vector[vector[double]]] c_neighbor_distss
        cdef vector[short] c_include_atoms

        if include_atoms is None:
            (<cppBasePotential*>self.thisptr.get()).get_neighbors(
                coords, c_neighbor_indss, c_neighbor_distss,
                cutoff_factor)
        else:
            c_include_atoms = vector[short](len(include_atoms))
            for i in xrange(len(include_atoms)):
                c_include_atoms[i] = include_atoms[i]
            (<cppBasePotential*>self.thisptr.get()).get_neighbors_picky(
                coords, c_neighbor_indss, c_neighbor_distss,
                c_include_atoms, cutoff_factor)

        neighbor_indss = []
        for i in xrange(c_neighbor_indss.size()):
            neighbor_indss.append([])
            for c_neighbor_ind in c_neighbor_indss[i]:
                neighbor_indss[-1].append(c_neighbor_ind)

        neighbor_distss = []
        for i in xrange(c_neighbor_distss.size()):
            neighbor_distss.append([])
            for c_nneighbor_dist in c_neighbor_distss[i]:
                neighbor_distss[-1].append([])
                for dist_comp in c_nneighbor_dist:
                    neighbor_distss[-1][-1].append(dist_comp)

        return neighbor_indss, neighbor_distss
