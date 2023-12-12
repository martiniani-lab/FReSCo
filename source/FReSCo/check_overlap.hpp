#ifndef _FRESCO_CHECK_OVERLAP_HPP
#define _FRESCO_CHECK_OVERLAP_HPP

#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "distance.hpp"

namespace fresco {

class CheckOverlapInterface{
public:
    virtual ~CheckOverlapInterface(){}
    virtual std::vector<size_t> get_overlapping_particles(const std::vector<double> &trial_coords){
        throw std::runtime_error("Not implemented error");
    }
    virtual std::vector<size_t> get_overlapping_particles_ca(const std::vector<double> &trial_coords, std::vector<long> & changed_atoms){
        throw std::runtime_error("Not implemented error");
    }
};

/**
 * Test for overlap of the hard sphere cores
 */
template <typename DIST_POL>
class CheckOverlap : CheckOverlapInterface{
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    std::vector<double> m_radii;
    const size_t m_nparticles;
    std::shared_ptr<DIST_POL> m_dist;
public:
    ~CheckOverlap() {};
    CheckOverlap(const std::vector<double> & radii, std::shared_ptr<DIST_POL> dist)
        : m_radii(radii),
          m_nparticles(m_radii.size()),
          m_dist(dist)
    {
        if (m_dist == NULL) {
            throw std::runtime_error("CheckOverlap: distance uninitialised");
        }
        if (radii.size() == 0) {
            throw std::runtime_error("CheckOverlap: illegal input: radii");
        }
        static_assert(DIST_POL::_ndim > 0, "CheckOverlap: illegal input: distance policy");
    }

    std::unordered_set<size_t> get_overlapping_particles_uset(const std::vector<double> &trial_coords,
                                std::vector<long> * changed_atoms=NULL)
    {
        if (trial_coords.size() % m_ndim) {
            throw std::runtime_error("CheckOverlap::conf_test: illegal input");
        }
        if (trial_coords.size() / m_ndim != m_nparticles) {
            throw std::runtime_error("CheckOverlap::conf_test: illegal input");
        }
        std::unordered_set<size_t> overlapping_particles_uset;
        overlapping_particles_uset.reserve(m_radii.size()+1);
        double dr[m_ndim];
        if (changed_atoms && changed_atoms->size() > 0) {
            for (const long i : *changed_atoms) {
                const size_t i1 = m_ndim * i;
                for (size_t j = 0; j < m_nparticles; ++j) {
                    if (i != j) {
                        const size_t j1 = m_ndim * j;
                        m_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
                        const double dij2 = std::inner_product(dr, dr + m_ndim, dr, double(0));
                        const double tmp = (m_radii[i] + m_radii[j]);
                        if (dij2 < tmp * tmp) {
                            overlapping_particles_uset.insert(i);
                            overlapping_particles_uset.insert(j);
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < m_nparticles; ++i) {
                const size_t i1 = m_ndim * i;
                for (size_t j = i + 1; j < m_nparticles; ++j) {
                    const size_t j1 = m_ndim * j;
                    m_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
                    const double dij2 = std::inner_product(dr, dr + m_ndim, dr, double(0));
                    const double tmp = (m_radii[i] + m_radii[j]);
                    if (dij2 < tmp * tmp) {
                        overlapping_particles_uset.insert(i);
                        overlapping_particles_uset.insert(j);
                    }
                }
            }
        }
        return overlapping_particles_uset;
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
class CheckOverlapPeriodic : public CheckOverlap<fresco::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodic(const std::vector<double> & radii, const std::vector<double> & boxvec)
        : CheckOverlap< fresco::periodic_distance<ndim> >(radii,
                std::make_shared<fresco::periodic_distance<ndim> >(boxvec))
    {}
};

template <size_t ndim>
class CheckOverlapCartesian : public CheckOverlap<fresco::cartesian_distance<ndim> > {
public:
    CheckOverlapCartesian(const std::vector<double> & radii)
        : CheckOverlap<fresco::cartesian_distance<ndim> >(radii,
                std::make_shared<fresco::cartesian_distance<ndim> >())
    {}
};

} // namespace bv

#endif //
