"""
# distutils: language = C++
"""
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
from distance_enum import Distance


# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type

# use external c++ classes
cdef extern from "FReSCo/distance.hpp" namespace "fresco":
    cdef cppclass cppCartesianDistance "fresco::cartesian_distance"[ndim]:
        cppCartesianDistance() except +
        void get_rij(double *, double *, double *)
        void get_pair_distances(vector[double]&, vector[double]&)
        void get_pair_distances_vec(vector[double]&, vector[double]&)
    cdef cppclass cppPeriodicDistance "fresco::periodic_distance"[ndim]:
        cppPeriodicDistance(vector[double] box) except +
        void get_rij(double *, double *, double *)
        void get_pair_distances(vector[double]&, vector[double]&)
        void get_pair_distances_vec(vector[double]&, vector[double]&)

cpdef get_distance(np.ndarray[double] r1, np.ndarray[double] r2, int ndim, method, box=None):
    """
    Define the Python interface to the C++ distance implementation.
    Parameters
    ----------
    r1 : [float]
        Position of the first particle
    r2 : [float]
        Position of the second particle
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array[float], optional
        Box size
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method in Distance, \
           "Distance measurement method undefined. It should be a Distance Enum."

    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d
    cdef cppCartesianDistance[INT2] *dist_cart_2d
    cdef cppCartesianDistance[INT3] *dist_cart_3d

    # Define box in C
    cdef vector[double] c_box

    # Initialize data arrays in C
    cdef double *c_r1 = <double *>malloc(ndim * sizeof(double))
    cdef double *c_r2 = <double *>malloc(ndim * sizeof(double))
    cdef double *c_r_ij = <double *>malloc(ndim * sizeof(double))
    for i in range(ndim):
        c_r1[i] = r1[i]
        c_r2[i] = r2[i]

    # Calculate the distance
    if method is Distance.PERIODIC:

        # Get box size from the input parameters
        assert box is not None, "Required argument 'box' not defined."
        c_box = box
        if ndim == 2:
            dist_per_2d = new cppPeriodicDistance[INT2](c_box)
            dist_per_2d.get_rij(c_r_ij, c_r1, c_r2)
        else:
            dist_per_3d = new cppPeriodicDistance[INT3](c_box)
            dist_per_3d.get_rij(c_r_ij, c_r1, c_r2)
    else:
        if ndim == 2:
            dist_cart_2d = new cppCartesianDistance[INT2]()
            dist_cart_2d.get_rij(c_r_ij, c_r1, c_r2)
        else:
            dist_cart_3d = new cppCartesianDistance[INT3]()
            dist_cart_3d.get_rij(c_r_ij, c_r1, c_r2)

    # Copy results into Python object
    r_ij = np.empty(ndim)
    for i in xrange(ndim):
        r_ij[i] = c_r_ij[i]

    # Free memory
    free(c_r1)
    free(c_r2)
    free(c_r_ij)

    return r_ij

cpdef get_pair_distances(np.ndarray[double] coords, int ndim, method, box=None):
    """
    Define the Python interface to the C++ distance implementation.
    Parameters
    ----------
    coords : [float]
        Position of the particles
    paird : [float]
        Empty vector
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array[float], optional
        Box size
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method in Distance, \
           "Distance measurement method undefined. It should be a Distance Enum."
    cdef size_t natoms = coords.size / ndim
    cdef size_t npairs =  natoms * (natoms-1) / 2
    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d
    cdef cppCartesianDistance[INT2] *dist_cart_2d
    cdef cppCartesianDistance[INT3] *dist_cart_3d

    # Define box in C
    cdef vector[double] c_box
    cdef vector[double] c_paird = vector[double](npairs)
    cdef vector[double] c_coords = coords

    # Calculate the distance
    if method is Distance.PERIODIC:
        # Get box size from the input parameters
        assert box is not None, "Required argument 'box' not defined."
        c_box = box
        if ndim == 2:
            dist_per_2d = new cppPeriodicDistance[INT2](c_box)
            dist_per_2d.get_pair_distances(c_coords, c_paird)
        else:
            dist_per_3d = new cppPeriodicDistance[INT3](c_box)
            dist_per_3d.get_pair_distances(c_coords, c_paird)
    else:
        if ndim == 2:
            dist_cart_2d = new cppCartesianDistance[INT2]()
            dist_cart_2d.get_pair_distances(c_coords, c_paird)
        else:
            dist_cart_3d = new cppCartesianDistance[INT3]()
            dist_cart_3d.get_pair_distances(c_coords, c_paird)

    return np.asarray(c_paird, dtype='d')

cpdef get_pair_distances_vec(np.ndarray[double] coords, int ndim, method, box=None):
    """
    Define the Python interface to the C++ distance implementation.
    Parameters
    ----------
    coords : [float]
        Position of the particles
    paird : [float]
        Empty vector
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array[float], optional
        Box size
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method in Distance, \
           "Distance measurement method undefined. It should be a Distance Enum."
    cdef size_t natoms = coords.size / ndim
    cdef size_t dnpairs =  ndim * natoms * (natoms-1) / 2
    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d
    cdef cppCartesianDistance[INT2] *dist_cart_2d
    cdef cppCartesianDistance[INT3] *dist_cart_3d

    # Define box in C
    cdef vector[double] c_box
    cdef vector[double] c_paird = vector[double](dnpairs)
    cdef vector[double] c_coords = coords

    # Calculate the distance
    if method is Distance.PERIODIC:
        # Get box size from the input parameters
        assert box is not None, "Required argument 'box' not defined."
        c_box = box
        if ndim == 2:
            dist_per_2d = new cppPeriodicDistance[INT2](c_box)
            dist_per_2d.get_pair_distances_vec(c_coords, c_paird)
        else:
            dist_per_3d = new cppPeriodicDistance[INT3](c_box)
            dist_per_3d.get_pair_distances_vec(c_coords, c_paird)
    else:
        if ndim == 2:
            dist_cart_2d = new cppCartesianDistance[INT2]()
            dist_cart_2d.get_pair_distances_vec(c_coords, c_paird)
        else:
            dist_cart_3d = new cppCartesianDistance[INT3]()
            dist_cart_3d.get_pair_distances_vec(c_coords, c_paird)

    return np.asarray(c_paird, dtype='d')
