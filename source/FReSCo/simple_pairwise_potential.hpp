#ifndef FRESCO_SIMPLE_PAIRWISE_POTENTIAL_HPP
#define FRESCO_SIMPLE_PAIRWISE_POTENTIAL_HPP

#include <string>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <math.h>
#include "FReSCo/base_potential.hpp"
#include "FReSCo/distance.hpp"
#include <vector>

namespace fresco{

template<typename pairwise_interaction, typename distance_policy>
class SimplePairwisePotential: public BasePotential{
    static const size_t m_ndim = distance_policy::_ndim;
    public:
        std::shared_ptr<pairwise_interaction> m_interaction;
        std::shared_ptr<distance_policy> m_distance;
        const std::vector<double> m_radii;

        SimplePairwisePotential(std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> distance, const std::vector<double> radii)
        : m_interaction(interaction),
          m_distance(distance),
          m_radii(radii)
        {}

        double sum_radii(const size_t atom_i, const size_t atom_j) const {
            if(m_radii.size() == 0) {
                return 0;
            } else {
                return m_radii[atom_i] + m_radii[atom_j];
            }
        }

        virtual size_t get_ndim(){return m_ndim;}

        virtual double get_energy(const std::vector<double> & x){
            return add_energy(x);
        }

        virtual double get_energy_gradient(const std::vector<double> & x, std::vector<double> & grad)
        {
            std::fill(grad.begin(), grad.end(), 0);
            return add_energy_gradient(x, grad);
        }

        virtual double get_energy_gradient_hessian(const std::vector<double> & x, std::vector<double> & grad, std::vector<double> & hess)
        {
            std::fill(grad.begin(), grad.end(), 0);
            std::fill(hess.begin(), hess.end(), 0);
            return add_energy_gradient_hessian(x, grad, hess);
        }

        inline double add_energy(const std::vector<double> & x)
        {
            const size_t natoms = x.size() / m_ndim;
            if (m_ndim * natoms != x.size()) {
                throw std::runtime_error("x.size() is not divisible by the number of dimensions");
            }
            double e=0.;
            double dr[m_ndim];

            for (size_t atom_i=0; atom_i<natoms; ++atom_i) {
                size_t i1 = m_ndim * atom_i;
                for (size_t atom_j=0; atom_j<atom_i; ++atom_j) {
                    size_t j1 = m_ndim * atom_j;

                    m_distance->get_rij(dr, &x[i1], &x[j1]);
                    double r2 = 0;
                    #pragma unroll
                    for (size_t k=0; k<m_ndim; ++k) {
                        r2 += dr[k]*dr[k];
                    }

                    e += m_interaction->energy(r2, sum_radii(atom_i, atom_j));
                }
            }
            return e;
        }

        inline double add_energy_gradient(const std::vector<double> & x, std::vector<double> & grad)
        {
            const size_t natoms = x.size() / m_ndim;
            if (m_ndim * natoms != x.size()) {
                throw std::runtime_error("x.size() is not divisible by the number of dimensions");
            }
            if (grad.size() != x.size()) {
                throw std::runtime_error("grad must have the same size as x");
            }

            double e = 0.;
            double gij;
            double dr[m_ndim];

            for (size_t atom_i=0; atom_i<natoms; ++atom_i) {
                const size_t i1 = m_ndim * atom_i;
                for (size_t atom_j=0; atom_j<atom_i; ++atom_j) {
                    const size_t j1 = m_ndim * atom_j;

                    m_distance->get_rij(dr, &x[i1], &x[j1]);
                    double r2 = 0;
                    #pragma unroll
                    for (size_t k=0; k<m_ndim; ++k) {
                        r2 += dr[k]*dr[k];
                    }

                    e += m_interaction->energy_gradient(r2, &gij, sum_radii(atom_i, atom_j));
                    if (gij != 0) {
                        #pragma unroll
                        for (size_t k=0; k<m_ndim; ++k) {
                            dr[k] *= gij;
                            grad[i1+k] -= dr[k];
                            grad[j1+k] += dr[k];
                        }
                    }
                }
            }
            return e;
        }

        inline double add_energy_gradient_hessian( const std::vector<double> & x, std::vector<double> & grad, std::vector<double> & hess)
        {
            double hij, gij;
            double dr[m_ndim];
            const size_t N = x.size();
            const size_t natoms = x.size() / m_ndim;
            if (m_ndim * natoms != x.size()) {
                throw std::runtime_error("x.size() is not divisible by the number of dimensions");
            }
            if (x.size() != grad.size()) {
                throw std::invalid_argument("the gradient has the wrong size");
            }
            if (hess.size() != x.size() * x.size()) {
                throw std::invalid_argument("the Hessian has the wrong size");
            }

            double e = 0.;
            for (size_t atom_i=0; atom_i<natoms; ++atom_i) {
                size_t i1 = m_ndim * atom_i;
                for (size_t atom_j=0; atom_j<atom_i; ++atom_j){
                    size_t j1 = m_ndim * atom_j;

                    m_distance->get_rij(dr, &x[i1], &x[j1]);
                    double r2 = 0;
                    #pragma unroll
                    for (size_t k=0; k<m_ndim; ++k) {
                        r2 += dr[k]*dr[k];
                    }

                    e += m_interaction->energy_gradient_hessian(r2, &gij, &hij, sum_radii(atom_i, atom_j));

                    if (gij != 0) {
                        #pragma unroll
                        for (size_t k=0; k<m_ndim; ++k) {
                            grad[i1+k] -= gij * dr[k];
                            grad[j1+k] += gij * dr[k];
                        }
                    }

                    if (hij != 0) {
                        #pragma unroll
                        for (size_t k=0; k<m_ndim; ++k){
                            //diagonal block - diagonal terms
                            double Hii_diag = (hij+gij)*dr[k]*dr[k]/r2 - gij;
                            hess[N*(i1+k)+i1+k] += Hii_diag;
                            hess[N*(j1+k)+j1+k] += Hii_diag;
                            //off diagonal block - diagonal terms
                            double Hij_diag = -Hii_diag;
                            hess[N*(i1+k)+j1+k] = Hij_diag;
                            hess[N*(j1+k)+i1+k] = Hij_diag;
                            #pragma unroll
                            for (size_t l = k+1; l<m_ndim; ++l){
                                //diagonal block - off diagonal terms
                                double Hii_off = (hij+gij)*dr[k]*dr[l]/r2;
                                hess[N*(i1+k)+i1+l] += Hii_off;
                                hess[N*(i1+l)+i1+k] += Hii_off;
                                hess[N*(j1+k)+j1+l] += Hii_off;
                                hess[N*(j1+l)+j1+k] += Hii_off;
                                //off diagonal block - off diagonal terms
                                double Hij_off = -Hii_off;
                                hess[N*(i1+k)+j1+l] = Hij_off;
                                hess[N*(i1+l)+j1+k] = Hij_off;
                                hess[N*(j1+k)+i1+l] = Hij_off;
                                hess[N*(j1+l)+i1+k] = Hij_off;
                            }
                        }
                    }
                }
            }
            return e;
        }

};

}

#endif
