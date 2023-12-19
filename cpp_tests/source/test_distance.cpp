#include <random>
#include <numeric>
#include <iterator>

#include <gtest/gtest.h>

#include "fresco/distance.hpp"
#include "test_utils.hpp"

using fresco::periodic_distance;
using fresco::cartesian_distance;

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  ASSERT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

#define TEST_REPEAT 100
#define BOX_LENGTH 10.0

class DistanceTest :  public ::testing::Test {
public:

    std::vector<double> x2[TEST_REPEAT];
    std::vector<double> y2[TEST_REPEAT];
    std::vector<double> x3[TEST_REPEAT];
    std::vector<double> y3[TEST_REPEAT];
    std::vector<double> x42[TEST_REPEAT];
    std::vector<double> y42[TEST_REPEAT];

    virtual void SetUp()
    {
        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> dist(1, 2*BOX_LENGTH);

        for (size_t i_repeat = 0; i_repeat < TEST_REPEAT; i_repeat++)  {
            x2[i_repeat] = std::vector<double>(2);
            y2[i_repeat] = std::vector<double>(2);
            for (size_t j = 0; j < 2; ++j) {
                x2[i_repeat][j] = dist(gen);
                y2[i_repeat][j] = dist(gen);
            }
        }

        for (size_t i_repeat = 0; i_repeat < TEST_REPEAT; i_repeat++)  {
            x3[i_repeat] = std::vector<double>(3);
            y3[i_repeat] = std::vector<double>(3);
            for (size_t j = 0; j < 3; ++j) {
                x3[i_repeat][j] = dist(gen);
                y3[i_repeat][j] = dist(gen);
            }
        }

        for (size_t i_repeat = 0; i_repeat < TEST_REPEAT; i_repeat++)  {
            x42[i_repeat] = std::vector<double>(42);
            y42[i_repeat] = std::vector<double>(42);
            for (size_t j = 0; j < 42; ++j) {
                x42[i_repeat][j] = dist(gen);
                y42[i_repeat][j] = dist(gen);
            }
        }
    }
};


TEST_F(DistanceTest, CartesianDistanceNorm_Works)
{
    for(size_t i_repeat = 0; i_repeat < TEST_REPEAT; i_repeat++)  {
        double dx_p_2[2];
        double dx_p_3[3];
        double dx_p_42[42];
        cartesian_distance<2>().get_rij(dx_p_2, x2[i_repeat].data(), y2[i_repeat].data());
        cartesian_distance<3>().get_rij(dx_p_3, x3[i_repeat].data(), y3[i_repeat].data());
        cartesian_distance<42>().get_rij(dx_p_42, x42[i_repeat].data(), y42[i_repeat].data());
        double ds_p_2 = 0;
        double ds_p_3 = 0;
        double ds_p_42 = 0;
        // compute with std
        double dx2[2];
        double dx3[3];
        double dx42[42];
        for (size_t i = 0; i < 2; ++i) {
            dx2[i] = x2[i_repeat][i] - y2[i_repeat][i];
            ASSERT_DOUBLE_EQ(dx_p_2[i], dx2[i]);
            ds_p_2 += dx_p_2[i] * dx_p_2[i];
        }
        for (size_t i = 0; i < 3; ++i) {
            dx3[i] = x3[i_repeat][i] - y3[i_repeat][i];
            ASSERT_DOUBLE_EQ(dx_p_3[i], dx3[i]);
            ds_p_3 += dx_p_3[i] * dx_p_3[i];
        }
        for (size_t i = 0; i < 42; ++i) {
            dx42[i] = x42[i_repeat][i] - y42[i_repeat][i];
            ASSERT_DOUBLE_EQ(dx_p_42[i], dx42[i]);
            ds_p_42 += dx_p_42[i] * dx_p_42[i];
        }
        const double ds2 = std::inner_product(dx2, dx2 + 2, dx2, double(0));
        const double ds3 = std::inner_product(dx3, dx3 + 3, dx3, double(0));
        const double ds42 = std::inner_product(dx42, dx42 + 42, dx42, double(0));
        // compare norms
        ASSERT_DOUBLE_EQ(ds_p_2, ds2);
        ASSERT_DOUBLE_EQ(ds_p_3, ds3);
        ASSERT_DOUBLE_EQ(ds_p_42, ds42);
    }
}

TEST_F(DistanceTest, CartesianPairDistance2_Works)
{
    const size_t natoms = TEST_REPEAT;
    std::vector<double> x2all(natoms*2);
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j=0; j<2; ++j){
            x2all[2*i_repeat+j] = x2[i_repeat][j];
        }
    }
    std::vector<double> paird = fresco::cartesian_distance<2>().get_pair_distances(x2all);

    size_t idx = 0;
    double dr[2];
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j_repeat = 0; j_repeat < i_repeat; j_repeat++)  {
            cartesian_distance<2>().get_rij(dr, x2[i_repeat].data(), x2[j_repeat].data());
            double r2 = 0;
            for (size_t k=0; k<2; ++k) {
                r2 += dr[k]*dr[k];
            }
            ASSERT_DOUBLE_EQ(paird[idx], std::sqrt(r2));
            ++idx;
        }
    }
}

TEST_F(DistanceTest, CartesianPairDistance3_Works)
{
    const size_t natoms = TEST_REPEAT;
    std::vector<double> x3all(natoms*3);
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j=0; j<3; ++j){
            x3all[3*i_repeat+j] = x3[i_repeat][j];
        }
    }
    std::vector<double> paird = fresco::cartesian_distance<3>().get_pair_distances(x3all);

    size_t idx = 0;
    double dr[3];
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j_repeat = 0; j_repeat < i_repeat; j_repeat++)  {
            cartesian_distance<3>().get_rij(dr, x3[i_repeat].data(), x3[j_repeat].data());
            double r2 = 0;
            for (size_t k=0; k<3; ++k) {
                r2 += dr[k]*dr[k];
            }
            ASSERT_DOUBLE_EQ(paird[idx], std::sqrt(r2));
            ++idx;
        }
    }
}

TEST_F(DistanceTest, CartesianPairDistance42_Works)
{
    const size_t natoms = TEST_REPEAT;
    std::vector<double> x42all(natoms*42);
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j=0; j<42; ++j){
            x42all[42*i_repeat+j] = x42[i_repeat][j];
        }
    }
    std::vector<double> paird = fresco::cartesian_distance<42>().get_pair_distances(x42all);

    size_t idx = 0;
    double dr[42];
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j_repeat = 0; j_repeat < i_repeat; j_repeat++)  {
            cartesian_distance<42>().get_rij(dr, x42[i_repeat].data(), x42[j_repeat].data());
            double r2 = 0;
            for (size_t k=0; k<42; ++k) {
                r2 += dr[k]*dr[k];
            }
            ASSERT_DOUBLE_EQ(paird[idx], std::sqrt(r2));
            ++idx;
        }
    }
}

TEST_F(DistanceTest, NearestImageConvention_Works)
{
    std::vector<double> x_out_of_box2(2);
    std::vector<double> x_boxed_true2(2);
    std::vector<double> x_boxed_per2(2);
    std::vector<double> x_out_of_box3(3);
    std::vector<double> x_boxed_true3(3);
    std::vector<double> x_boxed_per3(3);
    std::vector<double> x_out_of_box42(42);
    std::vector<double> x_boxed_per42(42);
    // The following means that "in box" is in [-8, 8].
    const double L = 16;
    x_out_of_box2[0] = -10;
    x_out_of_box2[1] = 20;
    x_boxed_true2[0] = 6;
    x_boxed_true2[1] = 4;
    std::copy(x_out_of_box2.begin(), x_out_of_box2.end(), x_boxed_per2.begin());
    // for(auto x:xp2){
    //     std::cout<<x<<" ";
    // }
    // std::cout<<std::endl;
    periodic_distance<2>(std::vector<double>(2, L)).put_in_box(x_boxed_per2);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(x_boxed_per2[i], x_boxed_true2[i]);
    }
    x_out_of_box3[0] = -9;
    x_out_of_box3[1] = 8.25;
    x_out_of_box3[2] = 12.12;
    x_boxed_true3[0] = 7;
    x_boxed_true3[1] = -7.75;
    x_boxed_true3[2] = -3.88;
    std::copy(x_out_of_box3.begin(), x_out_of_box3.end(), x_boxed_per3.begin());
    periodic_distance<3>(std::vector<double>(3, L)).put_in_box(x_boxed_per3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(x_boxed_per3[i], x_boxed_true3[i]);
    }
    // Assert that putting in box is irrelevant for distances.
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> dist(-100, 100);
    for (size_t i = 0; i < 42; ++i) {
        x_out_of_box42[i] = dist(gen);
    }
    std::vector<double> ones(42, 1);
    double delta42[42];
    periodic_distance<42>(std::vector<double>(42, L)).get_rij(delta42, &*ones.begin(), x_out_of_box42.data());
    const double d2_42_before = std::inner_product(delta42, delta42 + 42, delta42, double(0));
    std::copy(x_out_of_box42.begin(), x_out_of_box42.end(), x_boxed_per42.begin());
    periodic_distance<42>(std::vector<double>(42, L)).put_in_box(x_boxed_per42);
    for (size_t i = 0; i < 42; ++i) {
        EXPECT_LE(x_boxed_per42[i], 0.5 * L);
        EXPECT_LE(-0.5 * L, x_boxed_per42[i]);
    }
    periodic_distance<42>(std::vector<double>(42, L)).get_rij(delta42, &*ones.begin(), x_boxed_per42.data());
    const double d2_42_after = std::inner_product(delta42, delta42 + 42, delta42, double(0));
    EXPECT_DOUBLE_EQ(d2_42_before, d2_42_after);
}

/** Check the periodic put_atom_in_box method at the box boundary
 */
TEST_F(DistanceTest, PeriodicPutAtomInBox_BoxBoundaryWorks)
{
    std::vector<double> boxvec(2, 10);
    double boxboundary = boxvec[0] * 0.5;

    for (int i = -20; i <= 20; i++) {
        for (int j = -20; j <= 20; j++) {
            std::vector<double> coords(2, 0);
            std::vector<double> new_coords(2, 0);
            coords[0] = i * boxboundary + j * std::numeric_limits<double>::epsilon();
            periodic_distance<2>(boxvec).put_atom_in_box(new_coords.data(), coords.data());
            periodic_distance<2>(boxvec).put_atom_in_box(coords.data());
            EXPECT_LE(new_coords[0], boxboundary);
            EXPECT_GE(new_coords[0], -boxboundary);
            EXPECT_LE(coords[0], boxboundary);
            EXPECT_GE(coords[0], -boxboundary);
        }
    }
}

TEST_F(DistanceTest, SimplePeriodicNorm_Works)
{
    std::vector<double> bv2(2, BOX_LENGTH);
    std::vector<double> bv3(3, BOX_LENGTH);
    std::vector<double> bv42(42, BOX_LENGTH);

    for(size_t i_repeat = 0; i_repeat < TEST_REPEAT; i_repeat++)  {

        // compute with periodic_distance
        double dx_periodic_2d[2];
        double dx_periodic_3d[3];
        double dx_periodic_42d[42];
        periodic_distance<2>(bv2).get_rij(dx_periodic_2d, x2[i_repeat].data(), y2[i_repeat].data());
        periodic_distance<3>(bv3).get_rij(dx_periodic_3d, x3[i_repeat].data(), y3[i_repeat].data());
        periodic_distance<42>(bv42).get_rij(dx_periodic_42d, x42[i_repeat].data(), y42[i_repeat].data());

        // compute directly and compare
        double dx;
        for (size_t i = 0; i < 2; ++i) {
            dx = x2[i_repeat][i] - y2[i_repeat][i];
            dx -= round(dx / BOX_LENGTH) * BOX_LENGTH;
            ASSERT_DOUBLE_EQ(dx_periodic_2d[i], dx);
        }
        for (size_t i = 0; i < 3; ++i) {
            dx = x3[i_repeat][i] - y3[i_repeat][i];
            dx -= round(dx / BOX_LENGTH) * BOX_LENGTH;
            ASSERT_DOUBLE_EQ(dx_periodic_3d[i], dx);
        }
        for (size_t i = 0; i < 42; ++i) {
            dx = x42[i_repeat][i] - y42[i_repeat][i];
            dx -= round(dx / BOX_LENGTH) * BOX_LENGTH;
            ASSERT_DOUBLE_EQ(dx_periodic_42d[i], dx);
        }
    }
}

TEST_F(DistanceTest, PeriodicPairDistance2_Works)
{
    const size_t natoms = TEST_REPEAT;
    const double L = BOX_LENGTH;
    std::vector<double> x2all(natoms*2);
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j=0; j<2; ++j){
            x2all[2*i_repeat+j] = x2[i_repeat][j];
        }
    }
    std::vector<double> paird = fresco::periodic_distance<2>(std::vector<double>(2, L)).get_pair_distances(x2all);

    size_t idx = 0;
    double dr[2];
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j_repeat = 0; j_repeat < i_repeat; j_repeat++)  {
            periodic_distance<2>(std::vector<double>(2, L)).get_rij(dr, x2[i_repeat].data(), x2[j_repeat].data());
            double r2 = 0;
            for (size_t k=0; k<2; ++k) {
                r2 += dr[k]*dr[k];
            }
            ASSERT_DOUBLE_EQ(paird[idx], std::sqrt(r2));
            ++idx;
        }
    }
}

TEST_F(DistanceTest, PeriodicPairDistance3_Works)
{
    const size_t natoms = TEST_REPEAT;
    const double L = BOX_LENGTH;
    std::vector<double> x3all(natoms*3);
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j=0; j<3; ++j){
            x3all[3*i_repeat+j] = x3[i_repeat][j];
        }
    }
    std::vector<double> paird = fresco::periodic_distance<3>(std::vector<double>(3, L)).get_pair_distances(x3all);

    size_t idx = 0;
    double dr[3];
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j_repeat = 0; j_repeat < i_repeat; j_repeat++)  {
            periodic_distance<3>(std::vector<double>(3, L)).get_rij(dr, x3[i_repeat].data(), x3[j_repeat].data());
            double r2 = 0;
            for (size_t k=0; k<3; ++k) {
                r2 += dr[k]*dr[k];
            }
            ASSERT_DOUBLE_EQ(paird[idx], std::sqrt(r2));
            ++idx;
        }
    }
}

TEST_F(DistanceTest, PeriodicPairDistance42_Works)
{
    const size_t natoms = TEST_REPEAT;
    const double L = BOX_LENGTH;
    std::vector<double> x42all(natoms*42);
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j=0; j<42; ++j){
            x42all[42*i_repeat+j] = x42[i_repeat][j];
        }
    }
    std::vector<double> paird = fresco::periodic_distance<42>(std::vector<double>(42, L)).get_pair_distances(x42all);

    size_t idx = 0;
    double dr[42];
    for(size_t i_repeat = 0; i_repeat < natoms; i_repeat++)  {
        for(size_t j_repeat = 0; j_repeat < i_repeat; j_repeat++)  {
            periodic_distance<42>(std::vector<double>(42, L)).get_rij(dr, x42[i_repeat].data(), x42[j_repeat].data());
            double r2 = 0;
            for (size_t k=0; k<42; ++k) {
                r2 += dr[k]*dr[k];
            }
            ASSERT_DOUBLE_EQ(paird[idx], std::sqrt(r2));
            ++idx;
        }
    }
}

// /** Calculates the norm of a vector.
//  */
// template<typename dtype, size_t length>
// dtype norm(const dtype (&vec)[length]) {
//     dtype sum = 0;
//     for(const dtype elem : vec) {
//         sum += elem * elem;
//     }
//     return sum;
// }
