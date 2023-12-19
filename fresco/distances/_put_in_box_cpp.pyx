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
cdef extern from "fresco/distance.hpp" namespace "fresco":
    cdef cppclass cppPeriodicDistance "fresco::periodic_distance"[ndim]:
        cppPeriodicDistance(vector[double] box) except +
        void put_atom_in_box(double *)
        void put_in_box(vector[double]& double)


cpdef put_atom_in_box(np.ndarray[double] r, int ndim, method, np.ndarray[double] box):
    """
    Define the Python interface to the C++ distance implementation.
    Parameters
    ----------
    r : [float]
        Position of the particle
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array(float)
        Box size
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method is Distance.PERIODIC, \
           "Distance measurement method should be PERIODIC."

    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d

    # Initialize data arrays in C
    cdef double *c_r = <double *>malloc(ndim * sizeof(double))
    for i in xrange(ndim):
        c_r[i] = r[i]

    # Get box size from the input parameters
    cdef vector[double] c_box = box

    # Calculate the distance
    if ndim == 2:
        dist_per_2d = new cppPeriodicDistance[INT2](c_box)
        dist_per_2d.put_atom_in_box(c_r)
    else:
        dist_per_3d = new cppPeriodicDistance[INT3](c_box)
        dist_per_3d.put_atom_in_box(c_r)

    # Copy results into Python object
    r_boxed = np.empty(ndim)
    for i in xrange(ndim):
        r_boxed[i] = c_r[i]

    # Free memory
    free(c_r)

    return r_boxed


cpdef put_in_box(np.ndarray[double] rs, int ndim, method, np.ndarray[double] box):
    """
    Define the Python interface to the C++ distance implementation.
    Parameters
    ----------
    rs : np.array[float]
        Positions of all particles
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array[float]
        Box size
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method is Distance.PERIODIC, \
           "Distance measurement method should be PERIODIC."

    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d

    # Get box size from the input parameters
    cdef vector[double] c_box = box
    cdef vector[double] c_rs = rs

    # Calculate the distance
    if ndim == 2:
        dist_per_2d = new cppPeriodicDistance[INT2](c_box)
        dist_per_2d.put_in_box(c_rs)
    else:
        dist_per_3d = new cppPeriodicDistance[INT3](c_box)
        dist_per_3d.put_in_box(c_rs)

    # Copy results into Python object
    rs_boxed = np.empty(c_rs.size())
    for i in xrange(c_rs.size()):
        rs_boxed[i] = c_rs[i]

    return rs_boxed
