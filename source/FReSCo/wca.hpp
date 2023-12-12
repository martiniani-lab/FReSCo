#ifndef _FRESCO_WCA_HPP
#define _FRESCO_WCA_HPP

#include "FReSCo/simple_pairwise_potential.hpp"
#include "FReSCo/cell_list_potential.hpp"
#include "FReSCo/distance.hpp"
#include "FReSCo/base_interaction.hpp"

namespace fresco {

/**
 * Pairwise interaction for Weeks-Chandler-Andersen (WCA) potential
 */
struct WCA_interaction : BaseInteraction {
    double const _C6, _C12;
    double const _6C6, _12C12, _42C6, _156C12;
    double const _coff2, _eps; //cutoff distance for WCA potential

    WCA_interaction(double sig, double eps)
        : _C6(sig*sig*sig*sig*sig*sig),
          _C12(_C6*_C6), _6C6(6.*_C6),
          _12C12(12.*_C12), _42C6(42*_C6),
          _156C12(156*_C12),
          _coff2(pow(2.,1./3)*sig*sig), _eps(eps)
    {}

    /* calculate energy from distance squared */
    double energy(double r2, const double radius_sum) const
    {
        double E;
        double ir2 = 1.0/r2;
        double ir6 = ir2*ir2*ir2;
        double ir12 = ir6*ir6;
        if (r2 < _coff2)
            E = 4.*_eps*(-_C6*ir6 + _C12*ir12) + _eps;
        else
            E = 0.;

        return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
        double E;
        double ir2 = 1.0/r2;
        double ir6 = ir2*ir2*ir2;
        double ir12 = ir6*ir6;
        if (r2 < _coff2) {
            E = 4.*_eps*(-_C6*ir6 + _C12*ir12) + _eps;
            *gij = 4.*_eps*(- _6C6 * ir6 + _12C12 * ir12) * ir2;
        } else {
            E = 0.;
            *gij = 0;
        }

        return E;
    }

    double inline energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
        double E;
        double ir2 = 1.0/r2;
        double ir6 = ir2*ir2*ir2;
        double ir12 = ir6*ir6;

        if (r2 < _coff2) {
            E = 4.*_eps*(-_C6*ir6 + _C12*ir12) + _eps;
            *gij = 4.*_eps*(- _6C6 * ir6 + _12C12 * ir12) * ir2;
            *hij = 4.*_eps*(- _42C6 * ir6 + _156C12 * ir12) * ir2;
        } else {
            E = 0.;
            *gij = 0;
            *hij=0;
        }

        return E;
    }
};


//
// combine the components (interaction, looping method, distance function) into
// defined classes
//

/**
 * Pairwise WCA potential
 */
template<size_t ndim>
class WCA : public fresco::SimplePairwisePotential<fresco::WCA_interaction, fresco::cartesian_distance<ndim>> {
public:
    WCA(double sig, double eps, const std::vector<double> radii)
        : SimplePairwisePotential< fresco::WCA_interaction, fresco::cartesian_distance<ndim>>(
                std::make_shared<fresco::WCA_interaction>(sig, eps),
                std::make_shared<fresco::cartesian_distance<ndim>>(),
                radii)
    {}
};


/**
 * Pairwise WCA potential in a rectangular box
 */
template<size_t ndim>
class WCAPeriodic : public fresco::SimplePairwisePotential<fresco::WCA_interaction, fresco::periodic_distance<ndim> > {
public:
    WCAPeriodic(double sig, double eps, const std::vector<double> radii, const std::vector<double> boxvec)
        : SimplePairwisePotential<fresco::WCA_interaction, fresco::periodic_distance<ndim>> (
                std::make_shared<fresco::WCA_interaction>(sig, eps),
                std::make_shared<fresco::periodic_distance<ndim>>(boxvec),
                radii)
    {}
};



template <size_t ndim>
class WCAPeriodicCellLists : public CellListPotential< fresco::WCA_interaction, fresco::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    WCAPeriodicCellLists(double sigma, double eps,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPotential< fresco::WCA_interaction, fresco::periodic_distance<ndim> >
        (std::make_shared<fresco::WCA_interaction>(sigma, eps),
        std::make_shared<fresco::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv)
    {}

    

};
}
#endif
