#ifndef FRESCO_MOMENTS_H
#define FRESCO_MOMENTS_H

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>
#include <random>
#include <iostream>


namespace fresco{

class Moments {
protected:
    typedef double data_t;
    typedef size_t index_t;
private:
    data_t m_mean;
    data_t m_mean2;
    index_t m_count;
public:
    Moments(data_t mean=0, size_t count=0)
        : m_mean(mean),
          m_mean2(mean*mean),
          m_count(count)
    {}
    void update(const data_t input)
    {
        m_mean = (m_mean * m_count + input) / (m_count + 1);
        m_mean2 = (m_mean2 * m_count + (input * input)) / (m_count + 1);
//        if (m_count == std::numeric_limits<index_t>::max()) {
//            throw std::runtime_error("Moments: update: integer overflow");
//        }
        ++m_count;
    }

    void update(const data_t input, const size_t count)
    {
        m_mean = (m_mean * m_count + input * count) / (m_count + count);
        m_mean2 = (m_mean2 * m_count + (input * input) * count) / (m_count + count);
        m_count += count;
    }
    /**
     * replace a data point with another one
     */
    void replace(const data_t old_data, const data_t new_data)
    {
        m_mean += (new_data - old_data) / m_count;
        m_mean2 += (new_data * new_data - old_data * old_data) / m_count;
    }
    void operator() (const data_t input) { update(input); }
    index_t count() const { return m_count; }
    data_t mean() const { return m_mean; }
    data_t variance() const { return (m_mean2 - m_mean * m_mean); }
    data_t std() const { return sqrt(variance()); }

    void reset(){
        m_mean = 0;
        m_mean2 = 0;
        m_count = 0;
        }
};

class WeightedMoments {
protected:
    typedef double data_t;
private:
    data_t m_mean;
    data_t m_mean2;
    data_t m_sumw;
public:
    WeightedMoments(data_t mean=0, data_t sumw=0)
        : m_mean(mean),
          m_mean2(mean*mean),
          m_sumw(sumw)
    {}

    void update(const data_t input, const data_t w)
    {
        m_sumw += w;
        m_mean += (input - m_mean) * w / m_sumw;
        m_mean2 += (input * input - m_mean2) * w / m_sumw;
    }

    void operator() (const data_t input, const data_t w) { update(input, w); }
    data_t sum_weight() const { return m_sumw; }
    data_t mean() const { return m_mean; }
    data_t variance() const { return (m_mean2 - m_mean * m_mean); }
    data_t std() const { return sqrt(variance()); }

    void reset(){
        m_mean = 0;
        m_mean2 = 0;
        m_sumw = 0;
        }
};

template<class T>
std::vector<T> multiply(const std::vector<T>& a, const std::vector<T>& b){
    const size_t size = a.size();
    if (b.size() != size){throw std::runtime_error("multiply: vectors do not match in size");}
    std::vector<T> c(size);
    for (size_t i=0; i<size; ++i){
        c[i] = a[i] * b[i];
    }
    return c;
}

class VecMoments {
protected:
    typedef double data_t;
    typedef size_t index_t;
private:
    std::vector<data_t> m_mean;
    std::vector<data_t> m_mean2;
    std::vector<index_t> m_count;
    index_t m_vec_size;
public:
    VecMoments(const size_t vec_size)
        : m_mean(vec_size, 0),
          m_mean2(vec_size, 0),
          m_count(vec_size, 0),
          m_vec_size(vec_size)
    {}
    VecMoments(const std::vector<data_t>& input)
        : m_mean(input),
          m_mean2(multiply<data_t>(input, input)),
          m_count(input.size(), 0),
          m_vec_size(input.size())
    {}
    VecMoments(const std::vector<data_t>& means_vector, const std::vector<data_t>& counts_vector)
        : m_mean(means_vector),
          m_mean2(multiply<data_t>(means_vector, means_vector)),
          m_count(std::vector<index_t>(counts_vector.begin(), counts_vector.end())),
          m_vec_size(means_vector.size())
    {}

    VecMoments(const std::vector<data_t>& means_vector, const std::vector<data_t>& means2_vector, const std::vector<data_t>& counts_vector)
        : m_mean(means_vector),
          m_mean2(means2_vector),
          m_count(std::vector<index_t>(counts_vector.begin(), counts_vector.end())),
          m_vec_size(means_vector.size())
    {}

    void update(const std::vector<data_t>& input)
    {
        for (size_t i=0; i<m_vec_size; ++i){
            m_mean[i] = (m_mean[i] * m_count[i] + input[i]) / (m_count[i] + 1);
            m_mean2[i] = (m_mean2[i] * m_count[i] + (input[i] * input[i])) / (m_count[i] + 1);
            m_count[i] += 1;
        }
    }

    void update(const index_t i, const data_t input)
    {
        m_mean[i] = (m_mean[i] * m_count[i] + input) / (m_count[i] + 1);
        m_mean2[i] = (m_mean2[i] * m_count[i] + (input * input)) / (m_count[i] + 1);
        m_count[i] += 1;
    }

    void update(const std::vector<data_t>& input, const std::vector<size_t>& counts)
    {
        for (size_t i=0; i<m_vec_size; ++i){
            m_mean[i] = (m_mean[i] * m_count[i] + input[i] * counts[i]) / (m_count[i] + counts[i]);
            m_mean2[i] = (m_mean2[i] * m_count[i] + (input[i] * input[i]) * counts[i]) / (m_count[i] + counts[i]);
            m_count[i] += counts[i];
        }
    }

    /**
     * replace a data point with another one
     */
    void replace(const std::vector<data_t>& old_data, const std::vector<data_t>& new_data)
    {
        for (size_t i=0; i<m_vec_size; ++i){
            m_mean[i] += (new_data[i] - old_data[i]) / m_count[i];
            m_mean2[i] += (new_data[i] * new_data[i] - old_data[i] * old_data[i]) / m_count[i];
        }
    }

    void replace(const index_t i, const data_t old_data, const data_t new_data)
    {
        m_mean[i] += (new_data - old_data) / m_count[i];
        m_mean2[i] += (new_data * new_data - old_data * old_data) / m_count[i];
    }

    void operator() (const std::vector<data_t>& input) { update(input); }
    size_t size() const {return m_vec_size;}
    std::vector<index_t> count() const { return m_count; }
    std::vector<data_t> mean() const { return m_mean; }
    std::vector<data_t> variance() const {
        std::vector<data_t> var(m_vec_size);
        for (size_t i=0; i<m_vec_size; ++i){
            var[i] = m_mean2[i] - m_mean[i] * m_mean[i];
        }
        return var;
    }
    std::vector<data_t> std() const {
        std::vector<data_t> stdev(m_vec_size);
        for (size_t i=0; i<m_vec_size; ++i){
            stdev[i] = std::sqrt(m_mean2[i] - m_mean[i] * m_mean[i]);
        }
        return stdev;
    }
    void reset(){
        m_mean.assign(m_vec_size, 0);
        m_mean2.assign(m_vec_size, 0);
        m_count.assign(m_vec_size, 0);
    }
};

}

#endif // #ifndef
