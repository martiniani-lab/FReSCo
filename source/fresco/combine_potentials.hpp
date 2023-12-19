#ifndef _FRESCO_COMBINE_POTENTIALS_H_
#define _FRESCO_COMBINE_POTENTIALS_H_
#include "fresco/base_potential.hpp"
#include <cstddef>
#include <list>
#include <memory>

namespace fresco {

/**
 * Potential wrapper which wraps multiple potentials to
 * act as one potential.  This can be used to implement
 * multiple types of interactions, e.g. a system with two
 * types atoms
 */
class CombinedPotential : public BasePotential {
protected:
  std::list<std::shared_ptr<BasePotential>> _potentials;

public:
  CombinedPotential() {}

  /**
   * destructor: destroy all the potentials in the list
   */
  virtual ~CombinedPotential() {}

  /**
   * add a potential to the list
   */
  virtual void add_potential(std::shared_ptr<BasePotential> potential) {
    _potentials.push_back(potential);
  }

  virtual double get_energy(const std::vector<double> & x) {
    double energy = 0.;
    for (auto &pot_ptr : _potentials) {
      energy += pot_ptr->get_energy(x);
    }
    return energy;
  }

  virtual double get_energy_gradient(const std::vector<double> & x, 
                                     std::vector<double>& grad)
    {
    if (x.size() != grad.size()) {
      throw std::invalid_argument("the gradient has the wrong size");
    }

    double energy = 0.;
    grad.assign(grad.size(),0.);

    for (auto &pot_ptr : _potentials) {
      energy += pot_ptr->add_energy_gradient(x, grad);
    }
    return energy;
  }

};

} // namespace fresco

#endif
