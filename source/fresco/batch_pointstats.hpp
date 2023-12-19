#ifndef FRESCO_BATCH_PPS_HPP
#define FRESCO_BATCH_PPS_HPP

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

class BatchPPSInterface{
protected:
    typedef size_t index_t;
    typedef double data_t;
public:
    virtual ~BatchPPSInterface(){}
    virtual size_t get_seed(){throw std::runtime_error("Not implemented error");}
    virtual void set_generator_seed(const size_t inp){throw std::runtime_error("Not implemented error");}
    virtual std::vector<data_t> get_mean(){throw std::runtime_error("Not implemented error");}
    virtual std::vector<data_t> get_variance(){throw std::runtime_error("Not implemented error");}
    virtual std::vector<size_t> get_count(){throw std::runtime_error("Not implemented error");}
    virtual void run(const size_t n){throw std::runtime_error("Not implemented error");}
};


template <size_t ndim>
class BatchPPS : BatchPPSInterface {
protected:
    const static size_t m_ndim = ndim;
    const index_t m_natoms;
    size_t m_seed;
    std::vector<size_t> m_counts;
    std::vector<data_t> m_r2;
    std::mt19937_64 m_generator;
    std::vector<Moments> m_moments;
    fresco::periodic_distance<m_ndim> m_distance;
    const VecN<m_ndim, data_t> m_boxvec;
    std::vector<data_t> m_coords;
public:
    BatchPPS(std::vector<data_t> coords, std::vector<data_t> boxvec, std::vector<data_t> radii, const size_t rseed):
    m_natoms(coords.size() / m_ndim),
    m_seed(rseed),
    m_r2(radii),
    m_counts(radii.size()),
    m_generator(m_seed),
    m_moments(radii.size(),Moments()),
    m_distance(boxvec),
    m_boxvec(boxvec.begin(), boxvec.end()),
    m_coords(coords)
    {
        for (size_t i; i<m_r2.size(); i++){
            m_r2[i] *= m_r2[i];
        }
        // consistency checks
        assert(boxvec.size() == m_ndim);
        assert(m_coords.size() % m_ndim == 0);
        m_coords.shrink_to_fit();
    }

    void run(size_t n){
        // count the number of points within a distance i (assumes periodic boundary conditions)
        VecN<m_ndim, data_t> dr;
        size_t idx, step;
        VecN<m_ndim, data_t> centre;
        for (size_t x; x < n; x++)
        {
            for (size_t d=0; d<m_ndim; ++d){
                std::uniform_real_distribution<data_t> real_dist(0, m_boxvec[d]);
                centre[d] = real_dist(m_generator) - m_boxvec[d]/2;
            }
            m_counts.assign(m_counts.size(),0);
            #pragma omp parallel for
            for (size_t i=0; i<m_natoms; ++i){
                m_distance.get_rij(dr.data(), centre.data(), &m_coords[i*m_ndim]);
                data_t dr2 = dot<m_ndim, data_t>(dr, dr);
                if (dr2 <= m_r2[m_r2.size()-1]){
                    // Assumes m_r2 is ordered from smallest to largest
                    if (dr2 <= m_r2[0]){
                        m_counts[0] += 1;
                    }
                    else{
                        idx = size_t(m_counts.size()/2);
                        step = size_t(idx/2);
                        while (0==0){
                            if (dr2 <= m_r2[idx]){
                                if (dr2 > m_r2[idx-1]){
                                    m_counts[idx] += 1;
                                    break;
                                }
                                idx -= step;
                            }
                            else {
                                if (dr2 <= m_r2[idx+1]){
                                    m_counts[idx+1] += 1;
                                    break;
                                }
                                idx += step;
                            }
                            if (step>1){
                                step = size_t(step/2);
                            }
                        }
                    }
                }
            }
            int count = 0;
            for (size_t i=0; i<m_counts.size(); i++){
                //if (i<m_counts.size()-1)
                //    m_counts[i+1] += m_counts[i];
                ///m_moments[i](m_counts[i]);
                count += m_counts[i];
                m_moments[i](count);
            }
        }

    }

    std::vector<data_t> get_mean(){
        std::vector<data_t> means = std::vector<data_t>(m_moments.size());
        for (size_t i; i<m_moments.size();i++){
            means[i] = m_moments[i].mean();
        }
        return means;
    }

    std::vector<data_t> get_variance(){
        std::vector<data_t> vars = std::vector<data_t>(m_moments.size());
        for (size_t i; i<m_moments.size();i++){
            vars[i] = m_moments[i].variance();
        }
        return vars;
    }

    std::vector<size_t> get_count(){
        return m_counts;
    }

    size_t get_seed() const {return m_seed;}

    void set_generator_seed(const size_t inp) {
        m_seed = inp;
        m_generator.seed(m_seed);
        }
};

}

#endif // #ifndef
