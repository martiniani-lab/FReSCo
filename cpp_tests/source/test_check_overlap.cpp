#include <random>
#include <numeric>
#include <iterator>
#include <iostream>

#include <gtest/gtest.h>

#include "hyperalg/check_overlap.hpp"
#include "hyperalg/check_overlap_cell_lists.hpp"
#include "test_utils.hpp"

#if __APPLE__
#include "/usr/local/opt/libomp/include/omp.h"
#else
#include <omp.h>
#endif

using ha::periodic_distance;
using ha::cartesian_distance;

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  ASSERT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

#define NATOMS 1000
#define BOX_LENGTH 10.0
#define RADIUS 0.5


class CheckOverlapTest2d :  public ::testing::Test {
    static const size_t ndim = 2;
public:
    std::vector<double> x, radii, boxv;

    virtual void SetUp()
    {
        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> dist(0, BOX_LENGTH);
        radii = std::vector<double>(NATOMS, RADIUS);

        x = std::vector<double>(ndim*NATOMS);
        for (size_t j = 0; j < ndim*NATOMS; ++j) {
            x[j] = dist(gen);
        }

        boxv = std::vector<double>(ndim, BOX_LENGTH);

    }
};

TEST_F(CheckOverlapTest2d, CheckOverlapCartesian_Works)
{
    ha::CheckOverlapCartesian<2> check_overlap(radii);
    ha::CheckOverlapCartesianCellLists<2> check_overlap_cell_lists(radii, boxv);
    auto overlap_uset = check_overlap.get_overlapping_particles_uset(x);
    auto overlap_uset_cell = check_overlap_cell_lists.get_overlapping_particles_uset(x);
    bool test = overlap_uset == overlap_uset_cell;
    ASSERT_TRUE(test);
}

TEST_F(CheckOverlapTest2d, CheckOverlapPeriodic_Works)
{
    ha::CheckOverlapPeriodic<2> check_overlap(radii, boxv);
    ha::CheckOverlapPeriodicCellLists<2> check_overlap_cell_lists(radii, boxv);
    auto overlap_uset = check_overlap.get_overlapping_particles_uset(x);
    auto overlap_uset_cell = check_overlap_cell_lists.get_overlapping_particles_uset(x);
    bool test = overlap_uset == overlap_uset_cell;
    ASSERT_TRUE(test);
}

class CheckOverlapTest3d :  public ::testing::Test {
    static const size_t ndim = 3;
public:
    std::vector<double> x, radii, boxv;

    virtual void SetUp()
    {
        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> dist(1, BOX_LENGTH);
        radii = std::vector<double>(NATOMS, RADIUS);

        x = std::vector<double>(ndim*NATOMS);
        for (size_t j = 0; j < ndim*NATOMS; ++j) {
            x[j] = dist(gen);
        }

        boxv = std::vector<double>(ndim, BOX_LENGTH);

    }
};

TEST_F(CheckOverlapTest3d, CheckOverlapCartesian_Works)
{
    ha::CheckOverlapCartesian<3> check_overlap(radii);
    ha::CheckOverlapCartesianCellLists<3> check_overlap_cell_lists(radii, boxv);
    auto overlap_uset = check_overlap.get_overlapping_particles_uset(x);
    auto overlap_uset_cell = check_overlap_cell_lists.get_overlapping_particles_uset(x);
    bool test = overlap_uset == overlap_uset_cell;
    ASSERT_TRUE(test);
}

TEST_F(CheckOverlapTest3d, CheckOverlapPeriodic_Works)
{
    ha::CheckOverlapPeriodic<3> check_overlap(radii, boxv);
    ha::CheckOverlapPeriodicCellLists<3> check_overlap_cell_lists(radii, boxv);
    auto overlap_uset = check_overlap.get_overlapping_particles_uset(x);
    auto overlap_uset_cell = check_overlap_cell_lists.get_overlapping_particles_uset(x);
    bool test = overlap_uset == overlap_uset_cell;
    ASSERT_TRUE(test);
}
