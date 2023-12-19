#ifndef FRESCO_POINT_PATTERN_STATISTICS_HPP
#define FRESCO_POINT_PATTERN_STATISTICS_HPP

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <iterator>
#include <random>
#include <iostream>

#include "fresco/distance.hpp"
#include "fresco/moments.hpp"
#include "fresco/vecN.hpp"

namespace fresco{

class PointPatternStatisticsInterface{
protected:
    typedef size_t index_t;
    typedef double data_t;
public:
    virtual ~PointPatternStatisticsInterface(){}
    virtual size_t get_seed(){throw std::runtime_error("Not implemented error");}
    virtual void set_generator_seed(const size_t inp){throw std::runtime_error("Not implemented error");}
    virtual data_t get_mean(){throw std::runtime_error("Not implemented error");}
    virtual data_t get_variance(){throw std::runtime_error("Not implemented error");}
    virtual size_t get_count(){throw std::runtime_error("Not implemented error");}
    virtual void run(const size_t n){throw std::runtime_error("Not implemented error");}
};


template <size_t ndim>
class PointPatternStatistics : PointPatternStatisticsInterface {
protected:
    const static size_t m_ndim = ndim;
    const index_t m_natoms;
    size_t m_seed, m_count;
    const data_t m_radius;
    std::mt19937_64 m_generator;
    Moments m_moments;
    fresco::periodic_distance<m_ndim> m_distance;
    const VecN<m_ndim, data_t> m_boxvec;
    std::vector<data_t> m_coords;
public:
    PointPatternStatistics(std::vector<data_t> coords, std::vector<data_t> boxvec,
    const double radius, const size_t rseed):
    m_natoms(coords.size() / m_ndim),
    m_seed(rseed),
    m_radius(radius),
    m_generator(m_seed),
    m_moments(),
    m_distance(boxvec),
    m_boxvec(boxvec.begin(), boxvec.end()),
    m_coords(coords)
    {
        // consistency checks
        assert(boxvec.size() == m_ndim);
        assert(m_coords.size() % m_ndim == 0);
        m_coords.shrink_to_fit();
    }

    void run(size_t n){
        for (size_t i=0; i<n; ++i){
            take_sample();
        }
    }

    void take_sample(){
        //sample a point at random
        VecN<m_ndim, data_t> centre;
        for (size_t i=0; i<m_ndim; ++i){
            std::uniform_real_distribution<data_t> real_dist(0, m_boxvec[i]);
            centre[i] = real_dist(m_generator) - m_boxvec[i]/2;
        }
        // count the number of points within a distance i (assumes periodic boundary conditions)
        size_t n = 0;
        data_t r2 = m_radius*m_radius;
        VecN<m_ndim, data_t> dr;
        #pragma omp parallel for
        for (size_t i=0; i<m_natoms; ++i){
            m_distance.get_rij(dr.data(), centre.data(), &m_coords[i*m_ndim]);
            data_t dr2 = dot<m_ndim, data_t>(dr, dr);
            if (dr2 <= r2) {
                n += 1;
            }
        }
        m_moments(n);
    }

    double get_mean(){
        return m_moments.mean();
    }

    double get_variance(){
        return m_moments.variance();
    }

    size_t get_count(){
        return m_moments.count();
    }

    size_t get_seed() const {return m_seed;}

    void set_generator_seed(const size_t inp) {
        m_seed = inp;
        m_generator.seed(m_seed);
        }
};

}

#endif // #ifndef
