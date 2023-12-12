from FReSCo.potentials cimport _fresco
from libcpp.vector cimport vector
from FReSCo.potentials._fresco cimport shared_ptr
from libcpp cimport bool as cbool

cdef extern from "FReSCo/optimizer.hpp" namespace "fresco":
    cdef cppclass  cGradientOptimizer "fresco::GradientOptimizer":
        cGradientOptimizer(_fresco.cppBasePotential *, vector[double], double) except +
        void one_iteration() except+
        void run(int) except+
        void run() except+
        void set_func_gradient(double, vector[double]) except+
        void set_tol(double) except+
        void set_maxstep(double) except+
        void set_max_iter(int) except+
        void set_iprint(int) except+
        void set_verbosity(int) except+
        void reset(vector[double]&) except+
        vector[double] get_x() except+
        vector[double] get_g() except+
        double get_f() except+
        double get_rms() except+
        int get_nfev() except+
        int get_niter() except+
        int get_maxiter() except+
        cbool success() except+
        cbool stop_criterion_satisfied() except+
        void initialize_func_gradient() except+
    
cdef class GradientOptimizer:
    cdef shared_ptr[cGradientOptimizer] thisptr      # hold a C++ instance which we're wrapping
    cpdef events
