#ifndef _FRESCO_CHECK_OVERLAP_CELL_LISTS_H
#define _FRESCO_CHECK_OVERLAP_CELL_LISTS_H

#include "cell_lists.hpp"
#include "check_overlap.hpp"
#include <unordered_set>
#include <memory>
#include <algorithm>

namespace fresco {

template <class distance_policy>
class OverlapAccumulator {
//collect indexes of the overlapping particles
private:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<distance_policy> m_dist;
    std::shared_ptr<const std::vector<double>> m_coords;
    const std::vector<double> m_radii;
    std::vector<std::unordered_set<size_t>> m_overlapping_particles_usets;
public:
    OverlapAccumulator(std::shared_ptr<distance_policy> dist, const std::vector<double> & radii)
    : m_dist(dist),
    m_radii(radii)
    {
    #ifdef _OPENMP
    m_overlapping_particles_usets = std::vector<std::unordered_set<size_t>>(omp_get_max_threads());
    #pragma omp parallel
    {
       m_overlapping_particles_usets[omp_get_thread_num()].reserve((size_t) m_radii.size() / omp_get_max_threads());
    }
    #else
    m_overlapping_particles_usets = std::vector<std::unordered_set<size_t>>(1);
    m_overlapping_particles_usets[0].reserve(m_radii.size());
    #endif
    }

    void reset_data(std::shared_ptr<const std::vector<double>> coords) {
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
           m_overlapping_particles_usets[omp_get_thread_num()].clear();
           m_overlapping_particles_usets[omp_get_thread_num()].reserve((size_t) m_radii.size() / omp_get_max_threads());
        }
        #else
        m_overlapping_particles_usets[0].clear();
        m_overlapping_particles_usets[0].reserve(m_radii.size());
        #endif
    }

    std::unordered_set<size_t> get_overlapping_particels_uset() {
        #ifdef _OPENMP
        for (size_t i=1; i<omp_get_max_threads(); ++i){
            m_overlapping_particles_usets[0].insert(m_overlapping_particles_usets[i].begin(),
                                                    m_overlapping_particles_usets[i].end());
        }
        #endif
        return m_overlapping_particles_usets[0];
    }

    double get_squared_atom_distance(double const * const r1, double const * const r2) const
    {
        double dr[m_ndim];
        m_dist->get_rij(dr, r1, r2);
        return std::inner_product(dr, dr + m_ndim, dr, double(0));
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        const double dij2 = get_squared_atom_distance(m_coords->data() + xi_off, m_coords->data() + xj_off);
        const double radius_sum = m_radii[atom_i] + m_radii[atom_j];
        if(dij2 < radius_sum * radius_sum) {
            #ifdef _OPENMP
            m_overlapping_particles_usets[isubdom].insert(atom_i);
            m_overlapping_particles_usets[isubdom].insert(atom_j);
            #else
            m_overlapping_particles_usets[0].insert(atom_i);
            m_overlapping_particles_usets[0].insert(atom_j);
            #endif
        }
    }
};

template <typename DIST_POL>
class CellListCheckOverlap : CheckOverlapInterface{
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    std::shared_ptr<DIST_POL> m_dist;
    const std::vector<double> m_radii;
    std::shared_ptr<fresco::CellLists<DIST_POL> > m_cell_lists;
    OverlapAccumulator<DIST_POL> m_overlap_acc;
public:
    ~CellListCheckOverlap() {};
    CellListCheckOverlap(const std::vector<double> & radii, std::shared_ptr<DIST_POL> dist,
    std::shared_ptr<fresco::CellLists<DIST_POL> > cell_lists)
        :   m_dist(dist),
            m_radii(radii),
            m_cell_lists(cell_lists),
            m_overlap_acc(m_dist, m_radii)
    {
        if (m_dist == NULL || m_cell_lists == NULL) {
            throw std::runtime_error("CellListCheckOverlap: distance or celliter uninitialised");
        }
        if (m_dist == NULL) {
            throw std::runtime_error("CellListCheckOverlap: distance uninitialised");
        }
        if (radii.size() == 0) {
            throw std::runtime_error("CellListCheckOverlap: illegal input: radii");
        }
        static_assert(DIST_POL::_ndim > 0, "CellListCheckOverlap: illegal input: distance policy");
    }

    std::unordered_set<size_t> get_overlapping_particles_uset(const std::vector<double> & trial_coords,
                                const std::vector<long> * changed_atoms=NULL,
                                const std::vector<double> * changed_coords_old=NULL)
    {
        if (trial_coords.size() % m_ndim) {
            throw std::runtime_error("CellListCheckOverlap::conf_test: illegal input");
        }
        if (trial_coords.size() / m_ndim != m_radii.size()) {
            throw std::runtime_error("CellListCheckOverlap::conf_test: illegal input");
        }

        m_overlap_acc.reset_data(std::make_shared<const std::vector<double>>(trial_coords));
        if (changed_atoms && changed_coords_old && changed_atoms->size() > 0) {
            m_cell_lists->update_specific(trial_coords, *changed_atoms, *changed_coords_old);
            auto joe_the_looper = m_cell_lists->get_atom_pair_looper(m_overlap_acc);
            joe_the_looper.loop_through_atom_pairs_specific(trial_coords, *changed_atoms);
        } else {
            m_cell_lists->update(trial_coords);
            auto joe_the_looper = m_cell_lists->get_atom_pair_looper(m_overlap_acc);
            joe_the_looper.loop_through_atom_pairs();
        }

        return m_overlap_acc.get_overlapping_particels_uset();
    }

    size_t get_nr_cells(){
        return m_cell_lists->get_nr_cells();
    }

    std::vector<size_t> get_overlapping_particles_ca(const std::vector<double> &trial_coords, std::vector<long> & changed_atoms){
        auto overlapping_particles_uset = this->get_overlapping_particles_uset(trial_coords, &changed_atoms);
        std::vector<size_t> overlapping_particles(overlapping_particles_uset.begin(), overlapping_particles_uset.end());
        return overlapping_particles;
    }

    std::vector<size_t> get_overlapping_particles(const std::vector<double> &trial_coords){
        auto overlapping_particles_uset = this->get_overlapping_particles_uset(trial_coords);
        std::vector<size_t> overlapping_particles(overlapping_particles_uset.begin(), overlapping_particles_uset.end());
        return overlapping_particles;
    }
};

template <size_t ndim>
class CheckOverlapPeriodicCellLists : public CellListCheckOverlap<fresco::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodicCellLists(const std::vector<double> & radii, const std::vector<double> & boxvec, double ncellx_scale=1.0)
        : CellListCheckOverlap<fresco::periodic_distance<ndim> >(radii,
            std::make_shared<fresco::periodic_distance<ndim> >(boxvec),
            std::make_shared<fresco::CellLists<fresco::periodic_distance<ndim> > >(std::make_shared<fresco::periodic_distance<ndim> >(boxvec), boxvec, 2 * (*std::max_element(radii.begin(), radii.end())), ncellx_scale))
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellLists : public CellListCheckOverlap<fresco::cartesian_distance<ndim> > {
public:
    CheckOverlapCartesianCellLists(const std::vector<double> & radii, const std::vector<double> & boxvec, double ncellx_scale=1.0)
        : CellListCheckOverlap<fresco::cartesian_distance<ndim> >(radii,
                std::make_shared<fresco::cartesian_distance<ndim> >(),
                std::make_shared<fresco::CellLists<fresco::cartesian_distance<ndim> > >(std::make_shared<fresco::cartesian_distance<ndim> >(), boxvec, ncellx_scale))
    {}
};

} // namespace fresco

#endif // #ifndef
