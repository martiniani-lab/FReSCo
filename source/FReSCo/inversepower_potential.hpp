#ifndef FRESCO_INVERSEPOWER_HPP
#define FRESCO_INVERSEPOWER_HPP

#include <string>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <math.h>
#include "FReSCo/simple_pairwise_potential.hpp"
#include "FReSCo/cell_list_potential.hpp"
#include "FReSCo/distance.hpp"
#include "FReSCo/meta_pow.hpp"
#include "FReSCo/base_interaction.hpp"
#include <vector>

namespace fresco{

class InversePowerInteraction : public fresco::BaseInteraction{
public:
    const double m_pow; // Inverse power exponent
    const double m_eps;

    InversePowerInteraction(double a, double eps)
    : m_pow(a),
      m_eps(eps)
    {}

    /* calculate energy from distance squared */
    virtual double energy(double r2, const double radius_sum) const
    {
      double E;
      if (r2 >= radius_sum * radius_sum) {
          E = 0.;
      }
      else {
          // Sqrt moved into else, based on previous comment by CPG.
          const double r = std::sqrt(r2);
          E = std::pow((1 -r/radius_sum), m_pow) * m_eps/m_pow;
      }
      return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    virtual double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
      double E;
      if (r2 >= radius_sum * radius_sum) {
          E = 0.;
          *gij = 0.;
      }
      else {
          const double r = std::sqrt(r2);
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          E =  factor / m_pow;
          *gij =  - factor / ((r-radius_sum)*r);
      }
      return E;
    }

    virtual double energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
      double E;
      if (r2 >= radius_sum * radius_sum) {
          E = 0.;
          *gij = 0;
          *hij=0;
      }
      else {
          const double r = std::sqrt(r2);
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          const double denom = 1.0 / (r-radius_sum);
          E =  factor / m_pow;
          *gij =  - factor * denom / r ;
          *hij = (m_pow-1) * factor * denom * denom;
      }
      return E;
    }
};

template<size_t ndim>
class InversePowerCartesian : public fresco::SimplePairwisePotential<fresco::InversePowerInteraction, fresco::cartesian_distance<ndim>>{
    public:
    InversePowerCartesian(double a, double eps, const std::vector<double> radii)
    : SimplePairwisePotential<fresco::InversePowerInteraction, fresco::cartesian_distance<ndim>>
    (std::make_shared<fresco::InversePowerInteraction>(a, eps),
    std::make_shared<fresco::cartesian_distance<ndim>>(),
    radii)
    {}
};

template<size_t ndim>
class InversePowerPeriodic : public fresco::SimplePairwisePotential<fresco::InversePowerInteraction, fresco::periodic_distance<ndim>>{
    public:
    const std::vector<double> m_boxv;
    InversePowerPeriodic(double a, double eps, const std::vector<double> radii, const std::vector<double> boxv)
    : SimplePairwisePotential<fresco::InversePowerInteraction, fresco::periodic_distance<ndim>>
    (std::make_shared<fresco::InversePowerInteraction>(a, eps),
    std::make_shared<fresco::periodic_distance<ndim>>(boxv),
    radii),
    m_boxv(boxv)
    {}
};

template <size_t ndim>
class InversePowerPeriodicCellLists : public CellListPotential< fresco::InversePowerInteraction, fresco::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    InversePowerPeriodicCellLists(double pow, double eps,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPotential< fresco::InversePowerInteraction, fresco::periodic_distance<ndim> >
        (std::make_shared<fresco::InversePowerInteraction>(pow, eps),
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
