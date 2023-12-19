#ifndef _FRESCO_CELL_LIST_POTENTIAL_HPP
#define _FRESCO_CELL_LIST_POTENTIAL_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>

#include "fresco/distance.hpp"
#include "fresco/cell_lists.hpp"
#include "fresco/vecN.hpp"

namespace fresco{

/**
 * class which accumulates the energy one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyAccumulator {
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;
    std::vector<double*> m_energies;

public:
    ~EnergyAccumulator()
    {
        for(auto & energy : m_energies) {
            delete energy;
        }
    }

    EnergyAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : m_interaction(interaction),
          m_dist(dist),
          m_radii(radii)
    {
        #ifdef _OPENMP
        m_energies = std::vector<double*>(omp_get_max_threads());
        #pragma omp parallel
        {
            m_energies[omp_get_thread_num()] = new double();
        }
        #else
        m_energies = std::vector<double*>(1);
        m_energies[0] = new double();
        #endif
    }

    void reset_data(const std::vector<double> * coords) {
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            *m_energies[omp_get_thread_num()] = 0;
        }
        #else
        *m_energies[0] = 0;
        #endif
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        fresco::VecN<m_ndim, double> dr;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }
        #ifdef _OPENMP
        *m_energies[isubdom] += m_interaction->energy(r2, radius_sum);
        #else
        *m_energies[0] += m_interaction->energy(r2, radius_sum);
        #endif
    }

    double get_energy() {
        double energy = 0;
        for(size_t i = 0; i < m_energies.size(); ++i) {
            energy += *m_energies[i];
        }
        return energy;
    }
};

/**
 * class which accumulates the energy and gradient one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyGradientAccumulator {
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;
    std::vector<double*> m_energies;

public:
    std::vector<double> * m_gradient;

    ~EnergyGradientAccumulator()
    {
        for(auto & energy : m_energies) {
            delete energy;
        }
    }

    EnergyGradientAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : m_interaction(interaction),
          m_dist(dist),
          m_radii(radii)
    {
        #ifdef _OPENMP
        m_energies = std::vector<double*>(omp_get_max_threads());
        #pragma omp parallel
        {
            m_energies[omp_get_thread_num()] = new double();
        }
        #else
        m_energies = std::vector<double*>(1);
        m_energies[0] = new double();
        #endif
    }

    void reset_data(const std::vector<double> * coords, std::vector<double> * gradient) {
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            *m_energies[omp_get_thread_num()] = 0;
        }
        #else
        *m_energies[0] = 0;
        #endif
        m_gradient = gradient;
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        fresco::VecN<m_ndim, double> dr;
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double gij;
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }
        #ifdef _OPENMP
        *m_energies[isubdom] += m_interaction->energy_gradient(r2, &gij, radius_sum);
        #else
        *m_energies[0] += m_interaction->energy_gradient(r2, &gij, radius_sum);
        #endif
        if (gij != 0) {
            for (size_t k = 0; k < m_ndim; ++k) {
                dr[k] *= gij;
                (*m_gradient)[xi_off + k] -= dr[k];
                (*m_gradient)[xj_off + k] += dr[k];
            }
        }
    }

    double get_energy() {
        double energy = 0;
        for(size_t i = 0; i < m_energies.size(); ++i) {
            energy += *m_energies[i];
        }
        return energy;
    }
};

/**
 * class which accumulates the energy, gradient, and Hessian one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyGradientHessianAccumulator {
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;
    std::vector<double*> m_energies;

public:
    std::vector<double> * m_gradient;
    std::vector<double> * m_hessian;

    ~EnergyGradientHessianAccumulator()
    {
        for(auto & energy : m_energies) {
            delete energy;
        }
    }

    EnergyGradientHessianAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : m_interaction(interaction),
          m_dist(dist),
          m_radii(radii)
    {
        #ifdef _OPENMP
        m_energies = std::vector<double*>(omp_get_max_threads());
        #pragma omp parallel
        {
            m_energies[omp_get_thread_num()] = new double();
        }
        #else
        m_energies = std::vector<double*>(1);
        m_energies[0] = new double();
        #endif
    }

    void reset_data(const std::vector<double> * coords, std::vector<double> * gradient, std::vector<double> * hessian) {
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            *m_energies[omp_get_thread_num()] = 0;
        }
        #else
        *m_energies[0] = 0;
        #endif
        m_gradient = gradient;
        m_hessian = hessian;
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        fresco::VecN<m_ndim, double> dr;
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double gij, hij;
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }
        #ifdef _OPENMP
        *m_energies[isubdom] += m_interaction->energy_gradient_hessian(r2, &gij, &hij, radius_sum);
        #else
        *m_energies[0] += m_interaction->energy_gradient_hessian(r2, &gij, &hij, radius_sum);
        #endif
        if (gij != 0) {
            for (size_t k = 0; k < m_ndim; ++k) {
                (*m_gradient)[xi_off + k] -= gij * dr[k];
                (*m_gradient)[xj_off + k] += gij * dr[k];
            }
        }
        //this part is copied from simple_pairwise_potential.h
        //(even more so than the rest)
        const size_t N = m_gradient->size();
        const size_t i1 = xi_off;
        const size_t j1 = xj_off;
        for (size_t k = 0; k < m_ndim; ++k) {
            //diagonal block - diagonal terms
            const double Hii_diag = (hij + gij) * dr[k] * dr[k] / r2 - gij;
            (*m_hessian)[N * (i1 + k) + i1 + k] += Hii_diag;
            (*m_hessian)[N * (j1 + k) + j1 + k] += Hii_diag;
            //off diagonal block - diagonal terms
            const double Hij_diag = -Hii_diag;
            (*m_hessian)[N * (i1 + k) + j1 + k] = Hij_diag;
            (*m_hessian)[N * (j1 + k) + i1 + k] = Hij_diag;
            for (size_t l = k + 1; l < m_ndim; ++l) {
                //diagonal block - off diagonal terms
                const double Hii_off = (hij + gij) * dr[k] * dr[l] / r2;
                (*m_hessian)[N * (i1 + k) + i1 + l] += Hii_off;
                (*m_hessian)[N * (i1 + l) + i1 + k] += Hii_off;
                (*m_hessian)[N * (j1 + k) + j1 + l] += Hii_off;
                (*m_hessian)[N * (j1 + l) + j1 + k] += Hii_off;
                //off diagonal block - off diagonal terms
                const double Hij_off = -Hii_off;
                (*m_hessian)[N * (i1 + k) + j1 + l] = Hij_off;
                (*m_hessian)[N * (i1 + l) + j1 + k] = Hij_off;
                (*m_hessian)[N * (j1 + k) + i1 + l] = Hij_off;
                (*m_hessian)[N * (j1 + l) + i1 + k] = Hij_off;
            }
        }
    }

    double get_energy() {
        double energy = 0;
        for(size_t i = 0; i < m_energies.size(); ++i) {
            energy += *m_energies[i];
        }
        return energy;
    }
};


/**
 * class which accumulates the energy one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class NeighborAccumulator {
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> m_coords;
    const std::vector<double> m_radii;
    const double m_cutoff_sca;
    const std::vector<short> m_include_atoms;

public:
    std::vector<std::vector<size_t>> m_neighbor_indss;
    std::vector<std::vector<std::vector<double>>> m_neighbor_distss;

    NeighborAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & coords,
            std::vector<double> const & radii,
            const double cutoff_sca,
            std::vector<short> const & include_atoms)
        : m_interaction(interaction),
          m_dist(dist),
          m_coords(coords),
          m_radii(radii),
          m_cutoff_sca(cutoff_sca),
          m_include_atoms(include_atoms),
          m_neighbor_indss(radii.size()),
          m_neighbor_distss(radii.size())
    {}

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        if (m_include_atoms[atom_i] && m_include_atoms[atom_j]) {
            std::vector<double> dr(m_ndim);
            std::vector<double> neg_dr(m_ndim);
            const size_t xi_off = m_ndim * atom_i;
            const size_t xj_off = m_ndim * atom_j;
            m_dist->get_rij(dr.data(), m_coords.data() + xi_off, m_coords.data() + xj_off);
            double r2 = 0;
            for (size_t k = 0; k < m_ndim; ++k) {
                r2 += dr[k] * dr[k];
                neg_dr[k] = -dr[k];
            }
            const double radius_sum = m_radii[atom_i] + m_radii[atom_j];
            const double r_S = m_cutoff_sca * radius_sum;
            const double r_S2 = r_S * r_S;
            if(r2 <= r_S2) {
                m_neighbor_indss[atom_i].push_back(atom_j);
                m_neighbor_indss[atom_j].push_back(atom_i);
                m_neighbor_distss[atom_i].push_back(dr);
                m_neighbor_distss[atom_j].push_back(neg_dr);
            }
        }
    }
};



/**
 * Potential to loop over the list of atom pairs generated with the
 * cell list implementation in cell_lists.h.
 * This should also do the cell list construction and refresh, such that
 * the interface is the same for the user as with SimplePairwise.
 */
template <typename pairwise_interaction, typename distance_policy>
class CellListPotential : public BasePotential {
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    const std::vector<double> m_radii;
    fresco::CellLists<distance_policy> m_cell_lists;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;

    EnergyAccumulator<pairwise_interaction, distance_policy> m_eAcc;
    EnergyGradientAccumulator<pairwise_interaction, distance_policy> m_egAcc;
    EnergyGradientHessianAccumulator<pairwise_interaction, distance_policy> m_eghAcc;
public:
    ~CellListPotential() {}
    CellListPotential(
            std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const & boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            const bool balance_omp=true)
        : m_radii(radii),
          m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
          m_interaction(interaction),
          m_dist(dist),
          m_eAcc(interaction, dist, m_radii),
          m_egAcc(interaction, dist, m_radii),
          m_eghAcc(interaction, dist, m_radii)
    {}

    double sum_radii(const size_t atom_i, const size_t atom_j) const {
        if(m_radii.size() == 0) {
            return 0;
        } else {
            return m_radii[atom_i] + m_radii[atom_j];
        }
    }

    virtual size_t get_ndim() { return m_ndim; }

    virtual double get_energy(std::vector<double> const & coords)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return NAN;
        }

        update_iterator(coords);
        m_eAcc.reset_data(&coords);
        auto looper = m_cell_lists.get_atom_pair_looper(m_eAcc);

        looper.loop_through_atom_pairs();

        return m_eAcc.get_energy();
    }

    virtual double get_energy_gradient(std::vector<double> const & coords, std::vector<double> & grad)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != grad.size()) {
            throw std::invalid_argument("the gradient has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(grad.begin(), grad.end(), NAN);
            return NAN;
        }

        update_iterator(coords);
        std::fill(grad.begin(), grad.end(), 0.);
        m_egAcc.reset_data(&coords, &grad);
        auto looper = m_cell_lists.get_atom_pair_looper(m_egAcc);

        looper.loop_through_atom_pairs();

        return m_egAcc.get_energy();
    }

    virtual double get_energy_gradient_hessian(std::vector<double> const & coords,
            std::vector<double> & grad, std::vector<double> & hess)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != grad.size()) {
            throw std::invalid_argument("the gradient has the wrong size");
        }
        if (hess.size() != coords.size() * coords.size()) {
            throw std::invalid_argument("the Hessian has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(grad.begin(), grad.end(), NAN);
            std::fill(hess.begin(), hess.end(), NAN);
            return NAN;
        }

        update_iterator(coords);
        std::fill(grad.begin(), grad.end(), 0.);
        std::fill(hess.begin(), hess.end(), 0.);
        m_eghAcc.reset_data(&coords, &grad, &hess);
        auto looper = m_cell_lists.get_atom_pair_looper(m_eghAcc);

        looper.loop_through_atom_pairs();

        return m_eghAcc.get_energy();
    }

    virtual void get_neighbors(std::vector<double> const & coords,
                                std::vector<std::vector<size_t>> & neighbor_indss,
                                std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                const double cutoff_factor = 1.0)
    {
        size_t natoms = coords.size() / m_ndim;
        std::vector<short> include_atoms(natoms, 1);
        get_neighbors_picky(coords, neighbor_indss, neighbor_distss, include_atoms, cutoff_factor);
    }

    virtual void get_neighbors_picky(std::vector<double> const & coords,
                                      std::vector<std::vector<size_t>> & neighbor_indss,
                                      std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                      std::vector<short> const & include_atoms,
                                      const double cutoff_factor = 1.0)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (natoms != include_atoms.size()) {
            throw std::runtime_error("include_atoms.size() is not equal to the number of atoms");
        }
        if (m_radii.size() == 0) {
            throw std::runtime_error("Can't calculate neighbors, because the "
                                     "used interaction doesn't use radii. ");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return;
        }

        update_iterator(coords);
        NeighborAccumulator<pairwise_interaction, distance_policy> accumulator(
            m_interaction, m_dist, coords, m_radii, (1) * cutoff_factor, include_atoms);
        auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

        looper.loop_through_atom_pairs();

        neighbor_indss = accumulator.m_neighbor_indss;
        neighbor_distss = accumulator.m_neighbor_distss;
    }

protected:
    void update_iterator(std::vector<double> const & coords)
    {
        m_cell_lists.update(coords);
    }
};



} //namespace fresco

#endif //#ifndef _FRESCO_CELL_LIST_POTENTIAL_HPP
