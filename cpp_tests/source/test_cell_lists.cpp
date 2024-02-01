#include "FResCo/vecN.hpp"
#include "fresco/cell_lists.hpp"
#include "fresco/distance.hpp"
#include "fresco/inversepower_potential.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>
#include <random>
#include <ctime>
#include <algorithm>
#include <vector>
#include <omp.h>

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

template<size_t ndim>
class stupid_counter {
private:
    std::vector<size_t> m_count;
public:
    stupid_counter()
    {
        #ifdef _OPENMP
        m_count = std::vector<size_t>(omp_get_max_threads(), 0);
        #else
        m_count = std::vector<size_t>(1, 0);
        #endif
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        #ifdef _OPENMP
        m_count[omp_get_thread_num()]++;
        #else
        m_count[0]++;
        #endif
    }

    double get_count() {
        return std::accumulate(m_count.begin(), m_count.end(), 0);
    }
 };

template<typename DIST_POL>
class overlap_counter {
private:
    const static size_t m_ndim = DIST_POL::_ndim;
    std::vector<size_t> m_count;
    std::vector<double> m_coords;
    std::vector<double> m_radii;
    std::shared_ptr<DIST_POL> m_dist;
public:
    overlap_counter(std::vector<double> const & coords, std::vector<double> const & radii, std::shared_ptr<DIST_POL> const & dist)
    : m_coords(coords), m_radii(radii), m_dist(dist)
    {
        #ifdef _OPENMP
        m_count = std::vector<size_t>(omp_get_max_threads(), 0);
        #else
        m_count = std::vector<size_t>(1, 0);
        #endif
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        double dr[m_ndim];
        m_dist->get_rij(dr, &m_coords[atom_i * m_ndim], &m_coords[atom_j * m_ndim]);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        const double tmp = (m_radii[atom_i] + m_radii[atom_j]);
        if (r2 < tmp * tmp) {
            #ifdef _OPENMP
            m_count[omp_get_thread_num()]++;
            #else
            m_count[0]++;
            #endif
        }
    }

    double get_count() {
        return std::accumulate(m_count.begin(), m_count.end(), 0);
    }
 };

template<typename distance_policy>
size_t get_nr_unique_pairs(std::vector<double> coords, fresco::CellLists<distance_policy> & cl)
{
    stupid_counter<distance_policy::_ndim> counter;
    cl.update(coords);
    auto looper = cl.get_atom_pair_looper(counter);
    looper.loop_through_atom_pairs();
    return counter.get_count();
}

template<typename distance_policy>
size_t get_direct_nr_unique_pairs(std::shared_ptr<distance_policy> dist,
        const double max_distance, std::vector<double> x)
{
    static const size_t m_ndim = distance_policy::_ndim;
    size_t nr_unique_pairs = 0;
    const size_t natoms = x.size() / m_ndim;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            double rij[m_ndim];
            const double* xi = x.data() + i * m_ndim;
            const double* xj = x.data() + j * m_ndim;
            dist->get_rij(rij, xi, xj);
            double r2 = 0;
            for (size_t k = 0; k < m_ndim; ++k) {
                r2 += rij[k] * rij[k];
            }
            nr_unique_pairs += (r2 <= (max_distance * max_distance));
        }
    }
    return nr_unique_pairs;
}


class CellListsTest : public ::testing::Test {
public:
    double pow, eps, etrue, etrue_r, rcut, sca;
    std::vector<double> x, g, gnum, radii, boxvec;
    void SetUp(){
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
    	pow = 2.5;
    	eps = 1;
    	x = std::vector<double>(9);
        x[0] = 0.1;
        x[1] = 0.2;
        x[2] = 0.3;
        x[3] = 0.44;
        x[4] = 0.55;
        x[5] = 1.66;
        x[6] = 0.88;
        x[7] = 1.1;
        x[8] = 2.49;
        radii = std::vector<double>(3);
        boxvec = std::vector<double>(3, 5);
        for (size_t j = 0; j < 3; ++j) {
            double center = 0;
            for (size_t k = 0; k < 3; ++k) {
                center += x[k * 3 + j] / double(3);
            }
            for (size_t k = 0; k < 3; ++k) {
                x[k * 3 + j] -= center;
            }
        }
        double f = 1.;
        radii[0] = .91 * f;
        radii[1] = 1.1 * f;
        radii[2] = 1.13 * f;
        etrue = 0.03493116137645523;
        g = std::vector<double>(x.size());
        gnum = std::vector<double>(x.size());
        sca = 1.2;
        rcut = 2 * (1 + sca) * *std::max_element(radii.begin(), radii.end());
    }
};

//test number of distinguishable pairs
TEST_F(CellListsTest, Number_of_neighbors){
    fresco::CellLists<> cell(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0]);
    fresco::CellLists<> cell2(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0], 1);
    fresco::CellLists<> cell3(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0], 4.2);
    fresco::CellLists<> cell4(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0], 5);
    size_t count = 3u;
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell));
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell2));
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell3));
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell4));
}

TEST_F(CellListsTest, Number_of_neighbors_Cartesian){
    fresco::CellLists<fresco::cartesian_distance<3> > cell(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0]);
    fresco::CellLists<fresco::cartesian_distance<3> > cell2(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0], 1);
    fresco::CellLists<fresco::cartesian_distance<3> > cell3(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0], 4.2);
    fresco::CellLists<fresco::cartesian_distance<3> > cell4(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0], 5);
    size_t count = 3u;
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell));
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell2));
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell3));
    ASSERT_EQ(count, get_nr_unique_pairs(x, cell4));
}

TEST_F(CellListsTest, NumberNeighborsDifferentRcut_Works){
    auto dist = std::make_shared<fresco::periodic_distance<3> >(boxvec);
    fresco::CellLists<> cell(dist, boxvec, boxvec[0]);
    fresco::CellLists<> cell2(dist, boxvec, boxvec[0], 1);
    fresco::CellLists<> cell3(dist, boxvec, boxvec[0], 4.2);
    fresco::CellLists<> cell4(dist, boxvec, boxvec[0], 5);
    size_t count_ref = get_direct_nr_unique_pairs(dist, boxvec[0], x);
    size_t count = get_nr_unique_pairs(x, cell);
    size_t count2 = get_nr_unique_pairs(x, cell2);
    size_t count3 = get_nr_unique_pairs(x, cell3);
    size_t count4 = get_nr_unique_pairs(x, cell4);
    ASSERT_EQ(3u, count_ref);
    ASSERT_EQ(count_ref, count);
    ASSERT_EQ(count_ref, count2);
    ASSERT_EQ(count_ref, count3);
    ASSERT_EQ(count_ref, count4);
}

// TEST_F(CellListsTest, NumberNeighborsDifferentRcut_WorksLeesEdwards){
//     for(double shear = 0.0; shear <= 1.0; shear += 0.01) {
//         auto dist = std::make_shared<fresco::leesedwards_distance<3> >(boxvec, shear);
//         fresco::CellLists<fresco::leesedwards_distance<3>> cell(dist, boxvec, boxvec[0]);
//         fresco::CellLists<fresco::leesedwards_distance<3>> cell2(dist, boxvec, boxvec[0], 1);
//         fresco::CellLists<fresco::leesedwards_distance<3>> cell3(dist, boxvec, boxvec[0], 4.2);
//         fresco::CellLists<fresco::leesedwards_distance<3>> cell4(dist, boxvec, boxvec[0], 5);
//         size_t count_ref = get_direct_nr_unique_pairs(dist, boxvec[0], x);
//         size_t count = get_nr_unique_pairs(x, cell);
//         size_t count2 = get_nr_unique_pairs(x, cell2);
//         size_t count3 = get_nr_unique_pairs(x, cell3);
//         size_t count4 = get_nr_unique_pairs(x, cell4);
//         ASSERT_EQ(3u, count_ref);
//         ASSERT_EQ(count_ref, count);
//         ASSERT_EQ(count_ref, count2);
//         ASSERT_EQ(count_ref, count3);
//         ASSERT_EQ(count_ref, count4);
//     }
// }

TEST_F(CellListsTest, NumberNeighborsDifferentRcut_WorksCartesian){
    auto dist = std::make_shared<fresco::cartesian_distance<3> >();
    fresco::CellLists<fresco::cartesian_distance<3> > cell(dist, boxvec, boxvec[0]);
    fresco::CellLists<fresco::cartesian_distance<3> > cell2(dist, boxvec, boxvec[0], 1);
    fresco::CellLists<fresco::cartesian_distance<3> > cell3(dist, boxvec, boxvec[0], 4.2);
    fresco::CellLists<fresco::cartesian_distance<3> > cell4(dist, boxvec, boxvec[0], 5);
    size_t count = get_direct_nr_unique_pairs(dist, boxvec[0], x);
    size_t count2 = get_nr_unique_pairs(x, cell2);
    size_t count3 = get_nr_unique_pairs(x, cell3);
    size_t count4 = get_nr_unique_pairs(x, cell4);
    ASSERT_EQ(3u, count);
    ASSERT_EQ(count, count2);
    ASSERT_EQ(count, count3);
    ASSERT_EQ(count, count4);
}

TEST_F(CellListsTest, getEnergy_nan){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    std::fill(x.begin(), x.end(), NAN);
    const double e = pot_cell.get_energy(x);
    ASSERT_TRUE(isnan(e));
}

TEST_F(CellListsTest, getEnergy_inf){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    std::fill(x.begin(), x.end(), INFINITY);
    const double e = pot_cell.get_energy(x);
    ASSERT_TRUE(isnan(e));
}

TEST_F(CellListsTest, getEnergyGradient_nan){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    std::fill(x.begin(), x.end(), NAN);
    std::vector<double> g(x.size());
    std::fill(g.begin(), g.end(), NAN);
    const double e = pot_cell.get_energy_gradient(x, g);
    ASSERT_TRUE(isnan(e));
    ASSERT_TRUE(std::all_of(g.begin(), g.end(),
                        [](double elem) { return isnan(elem); }
                            ));
}

TEST_F(CellListsTest, getEnergyGradient_inf){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    std::fill(x.begin(), x.end(), INFINITY);
    std::vector<double> g(x.size());
    std::fill(g.begin(), g.end(), INFINITY);
    const double e = pot_cell.get_energy_gradient(x, g);
    ASSERT_TRUE(isnan(e));
    ASSERT_TRUE(std::all_of(g.begin(), g.end(),
                        [](double elem) { return isnan(elem); }
                            ));
}

TEST_F(CellListsTest, getEnergyGradientHessian_nan){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    std::fill(x.begin(), x.end(), NAN);
    std::vector<double> g(x.size());
    std::fill(g.begin(), g.end(), NAN);
    std::vector<double> h(x.size() * x.size());
    std::fill(h.begin(), h.end(), NAN);
    const double e = pot_cell.get_energy_gradient_hessian(x, g, h);
    ASSERT_TRUE(isnan(e));
    ASSERT_TRUE(std::all_of(g.begin(), g.end(),
                        [](double elem) { return isnan(elem); }
                            ));
    ASSERT_TRUE(std::all_of(h.begin(), h.end(),
                        [](double elem) { return isnan(elem); }
                            ));
}

TEST_F(CellListsTest, getEnergyGradientHessian_inf){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    std::fill(x.begin(), x.end(), INFINITY);
    std::vector<double> g(x.size());
    std::fill(g.begin(), g.end(), INFINITY);
    std::vector<double> h(x.size() * x.size());
    std::fill(h.begin(), h.end(), INFINITY);
    const double e = pot_cell.get_energy_gradient_hessian(x, g, h);
    ASSERT_TRUE(isnan(e));
    ASSERT_TRUE(std::all_of(g.begin(), g.end(),
                        [](double elem) { return isnan(elem); }
                            ));
    ASSERT_TRUE(std::all_of(h.begin(), h.end(),
                        [](double elem) { return isnan(elem); }
                            ));
}

TEST_F(CellListsTest, Energy_Works){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, 1.0);
    fresco::InversePowerPeriodicCellLists<3> pot_cell2(pow, eps, radii, boxvec, 2.0);
    fresco::InversePowerPeriodicCellLists<3> pot_cell3(pow, eps, radii, boxvec, 3.0);
    fresco::InversePowerPeriodicCellLists<3> pot_cell4(pow, eps, radii, boxvec, 4.0);
    fresco::InversePowerPeriodic<3> pot(pow, eps, radii, boxvec);
    const double ecell = pot_cell.get_energy(x);
    const double ecell2 = pot_cell2.get_energy(x);
    const double ecell3 = pot_cell3.get_energy(x);
    const double ecell4 = pot_cell4.get_energy(x);
    const double etrue = pot.get_energy(x);
    ASSERT_NEAR(ecell, etrue, 1e-10);
    ASSERT_NEAR(ecell2, etrue, 1e-10);
    ASSERT_NEAR(ecell3, etrue, 1e-10);
    ASSERT_NEAR(ecell4, etrue, 1e-10);
}

TEST_F(CellListsTest, ChangeCoords_Works){
    fresco::InversePowerPeriodicCellLists<3> pot_cell(pow, eps, radii, boxvec, .1);
    fresco::InversePowerPeriodic<3> pot(pow, eps, radii, boxvec);
    double ecell = pot_cell.get_energy(x);
    double etrue = pot.get_energy(x);
    ASSERT_NEAR(ecell, etrue, 1e-10);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += i;
    }
    ecell = pot_cell.get_energy(x);
    etrue = pot.get_energy(x);
    ASSERT_NEAR(ecell, etrue, 1e-10);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += (i+4)*4;
    }
    ecell = pot_cell.get_energy(x);
    etrue = pot.get_energy(x);
    ASSERT_NEAR(ecell, etrue, 1e-10);
}

TEST_F(CellListsTest, EnergyGradient_AgreesWithNumerical){
    fresco::InversePowerPeriodic<3> pot_no_cells(pow, eps, radii, boxvec);
    const double etrue = pot_no_cells.get_energy(x);
    const size_t N = 3;
    std::vector<std::shared_ptr<fresco::InversePowerPeriodicCellLists<3> > > pot;
    for (size_t i = 0; i < N; ++i) {
        pot.push_back(std::make_shared<fresco::InversePowerPeriodicCellLists<3> >(
                pow, eps, radii, boxvec, 1 + i));
    }
    pot.swap(pot);
    std::vector<double> e(N, 0);
    std::vector<double> ecomp(N, 0);
    for (size_t i = 0; i < N; ++i) {
        e.at(i) = pot.at(i)->get_energy_gradient(x, g);
        ecomp.at(i) = pot.at(i)->get_energy(x);
        pot.at(i)->numerical_gradient(x, gnum, 1e-6);
        for (size_t k = 0; k < 6; ++k) {
            ASSERT_NEAR(g[k], gnum[k], 1e-6);
        }
    }
    for (size_t i = 0; i < N; ++i) {
        ASSERT_NEAR(e.at(i), ecomp.at(i), 1e-10);
        ASSERT_NEAR(e.at(i), etrue, 1e-10);
    }
}

TEST_F(CellListsTest, EnergyGradientHessian_AgreesWithNumerical){
    fresco::InversePowerPeriodic<3> pot_no_cells(pow, eps, radii, boxvec);
    const double etrue = pot_no_cells.get_energy(x);
    std::vector<double> g_no_cells(x.size()) ;
    std::vector<double> h_no_cells(x.size() * x.size());
    pot_no_cells.get_energy_gradient_hessian(x, g_no_cells, h_no_cells);
    for (size_t i = 0; i < 3; ++i) {
        fresco::InversePowerPeriodicCellLists<3> pot(pow, eps, radii, boxvec, 1.0 + i);
        std::vector<double> h(x.size() * x.size());
        std::vector<double> hnum(h.size());
        const double e = pot.get_energy_gradient_hessian(x, g, h);
        const double ecomp = pot.get_energy(x);
        pot.numerical_gradient(x, gnum);
        pot.numerical_hessian(x, hnum);
        EXPECT_NEAR(e, ecomp, 1e-10);
        EXPECT_NEAR(etrue, ecomp, 1e-10);
        for (size_t i = 0; i < g.size(); ++i) {
            ASSERT_NEAR(g[i], gnum[i], 1e-10);
            ASSERT_NEAR(g[i], g_no_cells[i], 1e-10);
        }
        for (size_t i = 0; i < h.size(); ++i) {
            ASSERT_NEAR(h[i], hnum[i], 1e-10);
            ASSERT_NEAR(h[i], h_no_cells[i], 1e-10);
        }
    }
}

class CellListsTestHomogeneous3D : public ::testing::Test {
public:
    size_t nparticles;
    size_t boxdim;
    double boxlength;
    std::vector<double> boxvec;
    size_t ndof;
    size_t seed;
    std::mt19937_64 generator;
    std::uniform_real_distribution<double> distribution;
    std::vector<double> x;
    void SetUp(){
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        nparticles = 200;
        boxdim = 3;
        boxlength = 42;
        boxvec = std::vector<double>(boxdim, boxlength);
        ndof = nparticles * boxdim;
        seed = 42;
        generator = std::mt19937_64(seed);
        distribution = std::uniform_real_distribution<double>(-0.5 * boxlength, 0.5 * boxlength);
        x = std::vector<double>(ndof);
        for (size_t i = 0; i < ndof; ++i) {
            x[i] = distribution(generator);
        }
    }
};

TEST_F(CellListsTestHomogeneous3D, GridAndSpacing_Works) {
    fresco::CellLists<> cell_one(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0]);
    cell_one.update(x);
    EXPECT_EQ(cell_one.get_nr_cells(), 1u);
    EXPECT_EQ(cell_one.get_nr_cellsx(), 1u);
    //std::cout << "nr_unique_pairs: one:\n" << get_nr_unique_pairs(x, cell_one) << "\n";
    fresco::CellLists<> cell_two(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0] / 2);
    cell_two.update(x);
    EXPECT_EQ(cell_two.get_nr_cells(), 8u);
    EXPECT_EQ(cell_two.get_nr_cellsx(), 2u);
    //std::cout << "nr_unique_pairs: two:\n" << get_nr_unique_pairs(x, cell_two) << "\n";
    fresco::CellLists<> cell_three(std::make_shared<fresco::periodic_distance<3> >(boxvec), boxvec, boxvec[0] / 3);
    cell_three.update(x);
    EXPECT_EQ(cell_three.get_nr_cells(), 27u);
    EXPECT_EQ(cell_three.get_nr_cellsx(), 3u);
}

// TEST_F(CellListsTestHomogeneous3D, GridAndSpacingLeesEdwards_Works) {
//     for(double shear = 0.0; shear <= 1.0; shear += 0.01) {
//         fresco::CellLists<fresco::leesedwards_distance<3> > cell_one(
//             std::make_shared<fresco::leesedwards_distance<3> >(boxvec, shear), boxvec, boxvec[0]);
//         cell_one.update(x);
//         EXPECT_EQ(cell_one.get_nr_cells(), 1u);
//         EXPECT_EQ(cell_one.get_nr_cellsx(), 1u);
//         //std::cout << "nr_unique_pairs: one:\n" << get_nr_unique_pairs(x, cell_one) << "\n";
//         fresco::CellLists<fresco::leesedwards_distance<3> > cell_two(
//             std::make_shared<fresco::leesedwards_distance<3> >(boxvec, shear), boxvec, boxvec[0] / 2);
//         cell_two.update(x);
//         EXPECT_EQ(cell_two.get_nr_cells(), 8u);
//         EXPECT_EQ(cell_two.get_nr_cellsx(), 2u);
//         //std::cout << "nr_unique_pairs: two:\n" << get_nr_unique_pairs(x, cell_two) << "\n";
//         fresco::CellLists<fresco::leesedwards_distance<3> > cell_three(
//             std::make_shared<fresco::leesedwards_distance<3> >(boxvec, shear), boxvec, boxvec[0] / 3);
//         cell_three.update(x);
//         EXPECT_EQ(cell_three.get_nr_cells(), 27u);
//         EXPECT_EQ(cell_three.get_nr_cellsx(), 3u);
//     }
// }

TEST_F(CellListsTestHomogeneous3D, GridAndSpacingCartesian_Works) {
    fresco::CellLists<fresco::cartesian_distance<3> > cell_one(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0]);
    EXPECT_EQ(cell_one.get_nr_cells(), 1u);
    EXPECT_EQ(cell_one.get_nr_cellsx(), 1u);
    //std::cout << "nr_unique_pairs: one:\n" << get_nr_unique_pairs(x, cell_one) << "\n";
    fresco::CellLists<fresco::cartesian_distance<3> > cell_two(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0] / 2);
    EXPECT_EQ(cell_two.get_nr_cells(), 8u);
    EXPECT_EQ(cell_two.get_nr_cellsx(), 2u);
    //std::cout << "nr_unique_pairs: two:\n" << get_nr_unique_pairs(x, cell_two) << "\n";
    fresco::CellLists<fresco::cartesian_distance<3> > cell_three(std::make_shared<fresco::cartesian_distance<3> >(), boxvec, boxvec[0] / 3);
    EXPECT_EQ(cell_three.get_nr_cells(), 27u);
    EXPECT_EQ(cell_three.get_nr_cellsx(), 3u);
}


class CellListsTestHomogeneous2D : public ::testing::Test {
public:
    size_t nparticles;
    size_t boxdim;
    double boxlength;
    std::vector<double> boxvec;
    size_t ndof;
    size_t seed;
    std::mt19937_64 generator;
    std::uniform_real_distribution<double> distribution;
    std::vector<double> x;
    void SetUp(){
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        nparticles = 200;
        boxdim = 2;
        boxlength = 42;
        boxvec = std::vector<double>(boxdim, boxlength);
        ndof = nparticles * boxdim;
        seed = 42;
        generator = std::mt19937_64(seed);
        distribution = std::uniform_real_distribution<double>(-0.5 * boxlength, 0.5 * boxlength);
        x = std::vector<double>(ndof);
        for (size_t i = 0; i < ndof; ++i) {
            x[i] = distribution(generator);
        }
    }
};

TEST_F(CellListsTestHomogeneous2D, GridAndSpacing_Works) {
    fresco::CellLists<fresco::periodic_distance<2> > cell_one(std::make_shared<fresco::periodic_distance<2> >(boxvec), boxvec, boxvec[0]);
    EXPECT_EQ(cell_one.get_nr_cells(), 1u);
    EXPECT_EQ(cell_one.get_nr_cellsx(), 1u);
    //std::cout << "nr_unique_pairs: one:\n" << get_nr_unique_pairs(x, cell_one) << "\n";
    fresco::CellLists<fresco::periodic_distance<2> > cell_two(std::make_shared<fresco::periodic_distance<2> >(boxvec), boxvec, boxvec[0] / 2);
    EXPECT_EQ(cell_two.get_nr_cells(), 4u);
    EXPECT_EQ(cell_two.get_nr_cellsx(), 2u);
    //std::cout << "nr_unique_pairs: two:\n" << get_nr_unique_pairs(x, cell_two) << "\n";
    fresco::CellLists<fresco::periodic_distance<2> > cell_three(std::make_shared<fresco::periodic_distance<2> >(boxvec), boxvec, boxvec[0] / 3);
    EXPECT_EQ(cell_three.get_nr_cells(), 9u);
    EXPECT_EQ(cell_three.get_nr_cellsx(), 3u);
}

TEST_F(CellListsTestHomogeneous2D, GridAndSpacingCartesian_Works) {
    fresco::CellLists<fresco::cartesian_distance<2> > cell_one(std::make_shared<fresco::cartesian_distance<2> >(), boxvec, boxvec[0]);
    EXPECT_EQ(cell_one.get_nr_cells(), 1u);
    EXPECT_EQ(cell_one.get_nr_cellsx(), 1u);
    //std::cout << "nr_unique_pairs: one:\n" << get_nr_unique_pairs(x, cell_one) << "\n";
    fresco::CellLists<fresco::cartesian_distance<2> > cell_two(std::make_shared<fresco::cartesian_distance<2> >(), boxvec, boxvec[0] / 2);
    EXPECT_EQ(cell_two.get_nr_cells(), 4u);
    EXPECT_EQ(cell_two.get_nr_cellsx(), 2u);
    //std::cout << "nr_unique_pairs: two:\n" << get_nr_unique_pairs(x, cell_two) << "\n";
    fresco::CellLists<fresco::cartesian_distance<2> > cell_three(std::make_shared<fresco::cartesian_distance<2> >(), boxvec, boxvec[0] / 3);
    EXPECT_EQ(cell_three.get_nr_cells(), 9u);
    EXPECT_EQ(cell_three.get_nr_cellsx(), 3u);
}


class LatticeNeighborsTest : public ::testing::Test {
public:
    static const size_t ndim = 3;
    typedef fresco::periodic_distance<ndim> dist_t;
    std::vector<double> boxvec;
    double rcut;
    std::vector<size_t> ncells_vec;
    std::shared_ptr<dist_t> dist;

    virtual void SetUp(){
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        boxvec = std::vector<double>(3, 10);
        boxvec[1] += 1;
        boxvec[2] += 2;
        rcut = 20.; // large rcut means all cells are neighbors
        dist = std::make_shared<dist_t> (boxvec);
        ncells_vec = std::vector<size_t>(ndim);
        ncells_vec[0] = 2;
        ncells_vec[1] = 4;
        ncells_vec[2] = 20;
    }
};

TEST_F(LatticeNeighborsTest, LargeRcut_Works)
{
    static size_t const ndim = 3;
    typedef fresco::periodic_distance<ndim> dist_t;

    fresco::LatticeNeighbors<dist_t> lattice(dist, boxvec, rcut, ncells_vec);

    size_t icell = 8+3;
    auto v = lattice.global_ind_to_cell_vec(icell);
    // std::cout << v << std::endl;
    ASSERT_EQ(icell, lattice.cell_vec_to_global_ind(v));

    icell = 47;
    v = lattice.global_ind_to_cell_vec(icell);
    // std::cout << v << std::endl;
    ASSERT_EQ(icell, lattice.cell_vec_to_global_ind(v));


    auto neibs = lattice.find_all_global_neighbor_inds(0);
    ASSERT_EQ(neibs.size(), lattice.m_ncells);

    // check there are no duplicates
    std::set<size_t> s(neibs.begin(), neibs.end());
    ASSERT_EQ(neibs.size(), s.size());

    std::vector< std::vector< std::array<std::vector<long>*, 2> > > pairs_inner(lattice.m_nsubdoms);
    std::vector< std::vector< std::array<std::vector<long>*, 2> > > pairs_boundary(lattice.m_nsubdoms);
    std::vector< std::vector<std::vector<long>> > cells(lattice.m_nsubdoms);
    for (size_t isubdom = 0; isubdom < lattice.m_nsubdoms; isubdom++) {
        cells[isubdom] = std::vector<std::vector<long>>(lattice.cell_vec_to_global_ind(ncells_vec) / lattice.m_nsubdoms);
    }
    size_t total_cells = 0;
    for (size_t subdom_ncell : lattice.m_subdom_ncells) {
        total_cells += subdom_ncell;
    }
    std::vector< std::vector<std::vector<long>*> > cell_neighbors(total_cells);
    for (std::vector<std::vector<long>*> neighbors : cell_neighbors) {
        neighbors = std::vector<std::vector<long>*>();
    }
    lattice.find_neighbor_pairs(pairs_inner, pairs_boundary, cell_neighbors, cells);
    size_t count_neighbors = 0;
    for (size_t isubdom = 0; isubdom < lattice.m_nsubdoms; isubdom++) {
        count_neighbors += pairs_inner[isubdom].size() + pairs_boundary[isubdom].size();
    }
    ASSERT_EQ(count_neighbors, lattice.m_ncells * (lattice.m_ncells+1)/2);

    fresco::VecN<ndim> xpos(0.1);
    size_t icell1, isubdom1;
    lattice.position_to_local_ind(xpos.data(), icell1, isubdom1);

    xpos[0] += 2.;
    xpos[1] += 3.4;
    xpos[2] += .9;
    size_t icell2, isubdom2;
    lattice.position_to_local_ind(xpos.data(), icell2, isubdom2);
    ASSERT_NE(icell1, icell2);
}

TEST_F(LatticeNeighborsTest, SmallRcut_Works2)
{
    static size_t const ndim = 3;
    typedef fresco::periodic_distance<ndim> dist_t;
    rcut = .1; // small rcut means only adjacent cells are neighbors

    fresco::LatticeNeighbors<dist_t> lattice(dist, boxvec, rcut, ncells_vec);

    auto neibs = lattice.find_all_global_neighbor_inds(0);
    ASSERT_EQ(neibs.size(), size_t(2*3*3));

    // check there are no duplicates
    std::set<size_t> s(neibs.begin(), neibs.end());
    ASSERT_EQ(neibs.size(), s.size());

    std::vector< std::vector< std::array<std::vector<long>*, 2> > > pairs_inner(lattice.m_nsubdoms);
    std::vector< std::vector< std::array<std::vector<long>*, 2> > > pairs_boundary(lattice.m_nsubdoms);
    std::vector< std::vector<std::vector<long>> > cells(lattice.m_nsubdoms);
    for (size_t isubdom = 0; isubdom < lattice.m_nsubdoms; isubdom++) {
        cells[isubdom] = std::vector<std::vector<long>>(lattice.cell_vec_to_global_ind(ncells_vec) / lattice.m_nsubdoms);
    }
    size_t total_cells = 0;
    for (size_t subdom_ncell : lattice.m_subdom_ncells) {
        total_cells += subdom_ncell;
    }
    std::vector< std::vector<std::vector<long>*> > cell_neighbors(total_cells);
    for (std::vector<std::vector<long>*> neighbors : cell_neighbors) {
        neighbors = std::vector<std::vector<long>*>();
    }
    lattice.find_neighbor_pairs(pairs_inner, pairs_boundary, cell_neighbors, cells);
    size_t count_neighbors = 0;
    for (size_t isubdom = 0; isubdom < lattice.m_nsubdoms; isubdom++) {
        count_neighbors += pairs_inner[isubdom].size() + pairs_boundary[isubdom].size();
    }
    ASSERT_EQ(count_neighbors, (neibs.size() - 1) * lattice.m_ncells / 2 + lattice.m_ncells );
}

TEST_F(LatticeNeighborsTest, NonPeriodic_Works2)
{
    static size_t const ndim = 3;
    double rcut = .1; // small rcut means only adjacent cells are neighbors
    typedef fresco::cartesian_distance<ndim> dist_t;
    auto cart_dist = std::make_shared<dist_t> ();

    fresco::LatticeNeighbors<dist_t> lattice(cart_dist, boxvec, rcut, ncells_vec);

    auto neibs = lattice.find_all_global_neighbor_inds(0);
    ASSERT_EQ(neibs.size(), size_t(2*2*2));

    // check there are no duplicates
    std::set<size_t> s(neibs.begin(), neibs.end());
    ASSERT_EQ(neibs.size(), s.size());

    neibs = lattice.find_all_global_neighbor_inds(2);
    ASSERT_EQ(neibs.size(), size_t(2*3*2));
}

TEST_F(LatticeNeighborsTest, positionToCellVec_BoxBoundaryWorks)
{
    static size_t const ndim = 3;
    typedef fresco::periodic_distance<ndim> dist_t;
    fresco::LatticeNeighbors<dist_t> lattice(dist, boxvec, rcut, ncells_vec);

    double boxboundary = boxvec[0] * 0.5;
    for (int i = -20; i <= 20; i++) {
        for (int j = -20; j <= 20; j++) {
            std::vector<double> coords(ndim, 0);
            coords[0] = i * boxboundary + j * std::numeric_limits<double>::epsilon();
            fresco::VecN<ndim, size_t> cell_vec = lattice.position_to_cell_vec(coords.data());
            EXPECT_LT(cell_vec[0], lattice.m_ncells_vec[0]);
        }
    }
}


class CellListsSpecificTest : public ::testing::Test {
public:
    size_t seed;
    std::mt19937_64 generator;
    std::uniform_real_distribution<double> distribution;
    size_t nparticles;
    size_t ndim;
    size_t ndof;
    double eps;
    double sca;
    double r_hs;
    std::vector<double> x;
    std::vector<double> radii;
    std::vector<double> boxvec;
    double rcut;
    virtual void SetUp(){
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        seed = 42;
        generator = std::mt19937_64(seed);
        distribution = std::uniform_real_distribution<double>(-1, 1);
        nparticles = 50;
        ndim = 2;
        ndof = nparticles * ndim;
        eps = 1;
        r_hs = 1;
        x = std::vector<double>(ndof);
        radii = std::vector<double>(nparticles);
        for (size_t i = 0; i < nparticles; ++i) {
            radii[i] = r_hs;
        }
        rcut = 2 * *std::max_element(radii.begin(), radii.end());

        // Order atoms like a stair
        std::vector<double> coords(2);
        size_t k = 0;
        for(int i = 0; i < nparticles; i++) {
            coords[k] += 1;
            x[2*i] = coords[0];
            x[2*i + 1] = coords[1];
            if(k == 0) {
                k = 1;
            } else {
                k = 0;
            }
        }
        boxvec = std::vector<double>(ndim, std::max<double>(fabs(*std::max_element(x.data(), x.data() + ndof)), fabs(*std::min_element(x.data(), x.data() + ndof))) + rcut);
    }

    void create_coords() {
        // Order atoms like a stair, with some random component
        std::vector<double> coords(2);
        size_t k = 0;
        for(int i = 0; i < nparticles; i++) {
            coords[k] += (1 + (0.6 + 0.5 * distribution(generator)) * sca) * 2 * r_hs;
            x[2*i] = coords[0];
            x[2*i + 1] = coords[1];
            if(k == 0) {
                k = 1;
            } else {
                k = 0;
            }
        }
        boxvec = std::vector<double>(ndim, std::max<double>(fabs(*std::max_element(x.data(), x.data() + ndof)), fabs(*std::min_element(x.data(), x.data() + ndof))) + rcut);
    }
};

template<typename distance_policy>
size_t get_neighbors(std::vector<double> & coords, std::vector<long> & iatoms, std::vector<double> & old_coords, fresco::CellLists<distance_policy> & cl)
{
    stupid_counter<distance_policy::_ndim> counter;
    cl.update_specific(coords, iatoms, old_coords);
    auto looper = cl.get_atom_pair_looper(counter);
    looper.loop_through_atom_pairs_specific(coords, iatoms);
    return counter.get_count();
}

TEST_F(CellListsSpecificTest, Number_of_neighbors){
    fresco::CellLists<fresco::periodic_distance<2>> cell(std::make_shared<fresco::periodic_distance<2> >(boxvec), boxvec, rcut);
    std::vector<long> iatoms(0);
    std::vector<double> old_coords(0);
    for (long i = 0; i < nparticles; ++i) {
        iatoms.push_back(i);
        for (size_t idim = 0; idim < ndim; ++idim) {
            old_coords.push_back(x[i * ndim + idim]);
        }
    }
    size_t neighbors = get_neighbors(x, iatoms, old_coords, cell);
    size_t pairs = get_nr_unique_pairs(x, cell);
    ASSERT_EQ(2 * pairs, neighbors);

    create_coords();
    neighbors = get_neighbors(x, iatoms, old_coords, cell);
    pairs = get_nr_unique_pairs(x, cell);
    ASSERT_EQ(2 * pairs, neighbors);
}

template<typename distance_policy>
size_t get_overlaps(
    std::vector<double> const & coords,
    std::vector<long> const & iatoms,
    std::vector<double> const & old_coords,
    std::vector<double> const & radii,
    fresco::CellLists<distance_policy> & cl,
    std::shared_ptr<distance_policy> const & dist)
{
    overlap_counter<distance_policy> counter(coords, radii, dist);
    cl.update_specific(coords, iatoms, old_coords);
    auto looper = cl.get_atom_pair_looper(counter);
    looper.loop_through_atom_pairs_specific(coords, iatoms);
    return counter.get_count();
}

TEST_F(CellListsSpecificTest, Number_of_overlaps){
    auto dist = std::make_shared<fresco::periodic_distance<2> >(boxvec);
    fresco::CellLists<fresco::periodic_distance<2>> cell(dist, boxvec, rcut);
    std::vector<long> iatoms(0);
    std::vector<double> old_coords(0);
    for (long i = 10; i < 40; ++i) {
        iatoms.push_back(i);
        for (size_t idim = 0; idim < ndim; ++idim) {
            old_coords.push_back(x[i * ndim + idim]);
        }
        ASSERT_EQ(4 * (i-9), get_overlaps(x, iatoms, old_coords, radii, cell, dist));
    }
}
