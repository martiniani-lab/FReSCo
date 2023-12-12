#include "hyperalg/base_potential.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>

using ha::BasePotential;

class HarmonicE : public ha::BasePotential
{
public:
    size_t call_count;
    HarmonicE() : call_count(0) {}

    virtual double get_energy(const std::vector<double> & x){
        call_count += 1;
        double energy = 0;
        for (size_t k=0; k<x.size(); ++k){
            energy += x[k] * x[k];
        }
        return energy / 2.;
    }
};

class BasePotentialTest :  public ::testing::Test
{
public:
    double etrue;
    std::vector<double> x, g, gtrue, hess, htrue;
    virtual void SetUp(){
        x = std::vector<double>(2);
        g = std::vector<double>(2);
        hess = std::vector<double>(2*2);
        htrue = std::vector<double>(4, 0.);
        x[0] = 1.;
        x[1] = 1.5;
        etrue = (x[0] * x[0] + x[1] * x[1]) / 2.;
        gtrue = x;
        htrue[0] = 1.;
        htrue[3] = 1.;
    }
};


TEST_F(BasePotentialTest, EOnlyEnergy_Works){
    HarmonicE pot;
    BasePotential * ptr = &pot;
    double e = pot.get_energy(x);
    EXPECT_EQ(1u, pot.call_count);
    EXPECT_NEAR(e, etrue, 1e-10);
    double e_ptr = ptr->get_energy(x);
    EXPECT_EQ(2u, pot.call_count);
    EXPECT_NEAR(e_ptr, etrue, 1e-10);
}

TEST_F(BasePotentialTest, EOnlyGrad_Works){
    HarmonicE pot;
    double e = pot.get_energy_gradient(x, g);
    EXPECT_NEAR(e, etrue, 1e-10);
    EXPECT_EQ(1+2*x.size(), pot.call_count);

    // the gradient is computed numerically
    for (size_t k=0; k<x.size(); ++k){
        EXPECT_NEAR(g[k], gtrue[k], 1e-6);
    }
}

TEST_F(BasePotentialTest, EOnlyHess_Works){
    HarmonicE pot;
    double e = pot.get_energy_gradient_hessian(x, g, hess);
    EXPECT_NEAR(e, etrue, 1e-10);
    size_t count_energy_grad = 1+2*x.size();
    size_t count_hess = 2*x.size()*count_energy_grad;
    EXPECT_EQ(count_energy_grad + count_hess, pot.call_count);


    // the gradient is computed numerically
    for (size_t k=0; k<x.size(); ++k){
        EXPECT_NEAR(g[k], gtrue[k], 1e-6);
    }

    // the hessian is computed numerically
    for (size_t k=0; k<hess.size(); ++k){
        EXPECT_NEAR(hess[k], htrue[k], 1e-3);
    }
}

TEST_F(BasePotentialTest, EOnlyGetHess_Works){
    HarmonicE pot;
    pot.get_hessian(x, hess);
    size_t count_energy_grad = 1+2*x.size();
    size_t count_hess = 2*x.size()*count_energy_grad;
    EXPECT_EQ(count_energy_grad + count_hess, pot.call_count);

    // the hessian is computed numerically
    for (size_t k=0; k<hess.size(); ++k){
        EXPECT_NEAR(hess[k], htrue[k], 1e-3);
    }
}

TEST_F(BasePotentialTest, Throws){
    BasePotential pot;
    EXPECT_THROW(pot.get_energy(x), std::runtime_error);
    EXPECT_THROW(pot.numerical_gradient(x, hess), std::invalid_argument);
    EXPECT_THROW(pot.numerical_hessian(hess, hess), std::invalid_argument);
}
