#include <string>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "finufft.h"
#include "base_potential.hpp"
#include <vector>
#include <complex>
#include <random>

//static const double M_PI = 3.1415926;
static const std::complex<double> I(0.0,1.0);

namespace fresco{

class UwNU: public BasePotential{
    //static const size_t ndim = 2;
    public:
        const size_t ndim; //dimension
        const size_t Nk; // Number K vectors
        std::vector<int64_t> N; // input grid size
        const std::vector<double> L; // box dimensions
        std::vector<double> kx; // K positions
        std::vector<double> ky; // K positions
        std::vector<double> kz; // K positions
        const double eps; // finufft error tolerance
        const std::vector<double> Sk0; // Structure Factor
        std::vector<std::complex<double>> c; // complex weighting
        const std::vector<double> V; // Potential Weighting
        finufft_plan plan1;
        finufft_plan plan2;

        UwNU(std::vector<int> _N, std::vector<double> _K, std::vector<double> _Sk, std::vector<double> _V, std::vector<double> _L, double _eps)
        : L(_L),
          ndim(_L.size()),
          N(ndim),
          Sk0(_Sk),
          Nk(_Sk.size()),
          V(_V),
          eps(_eps),
          kx(Nk)
        {
            N[0] = int64_t(_N[0]);
            size_t Ni = size_t(_N[0]);
	    for (size_t j = 0; j < Nk; j++)
            {
	        kx[j] = _K[ndim*j]*2*M_PI/L[0];
            }
            if (ndim > 1)
            {
                N[1] = int64_t(_N[1]);
                Ni *= size_t(_N[1]);
                ky = std::vector<double>(Nk); // y coordinate
                #pragma omp parallel for
	        for (size_t j = 0; j < Nk; j++)
	        {
                    ky[j] = _K[ndim*j+1]*2*M_PI/L[1];
                }
            }
            else
            {
                ky = std::vector<double>(1);
            }
            if (ndim > 2)
            {
                N[2] = int64_t(_N[2]);
                Ni *= size_t(_N[2]);
                kz = std::vector<double>(Nk); // z coordinate
                #pragma omp parallel for
	        for (size_t j = 0; j < Nk; j++)
	        {
                    kz[j] = _K[ndim*j+2]*2*M_PI/L[2];
                }
            }
            else
            {
                kz = std::vector<double>(1);
            }
            c = std::vector<std::complex<double>>(Ni);
            finufft_makeplan(1, ndim, &N[0], -1, 1, eps, &plan1, NULL);
            finufft_makeplan(2, ndim, &N[0], +1, 1, eps, &plan2, NULL);
            finufft_setpts(plan1, Nk, &kx[0], &ky[0], &kz[0], 0, &kx[0], &ky[0], &kz[0]);
            finufft_setpts(plan2, Nk, &kx[0], &ky[0], &kz[0], 0, &kx[0], &ky[0], &kz[0]);
        }

        virtual double get_energy(const std::vector<double>& points)
        {
	        std::vector<std::complex<double>> rho(Nk), c(points.size());
            double rhotot = 1.0;
            double Skdiff, Skdiff2;
            double phi = 0.0;
            for (size_t j = 0; j < points.size(); j++)
            {
                c[j] = points[j];
                //rhotot += points[j];
            }

            finufft_execute(plan2, &rho[0], &c[0]);
            
            for (size_t i = 0; i < rho.size(); i++)
            {
                Skdiff = std::real(std::abs(rho[i]));
                std::cout << Skdiff << '\n'; 
                Skdiff = Skdiff*Skdiff/rhotot-Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
             
                Skdiff2 = Skdiff*Skdiff;
                phi += V[i]*Skdiff2;
            } 
            return phi;
        }

        virtual double get_energy_gradient(const std::vector<double>& points, std::vector<double>& grad)
        {
	        std::vector<std::complex<double>> rho(Nk), f(Nk), c(points.size());
            double rhotot = 1.0;
            double Skdiff, Skdiff2;
            double phi = 0.0;
	        grad.assign(grad.size(),0);
            for (size_t j = 0; j < points.size(); j++)
            {
                c[j] = points[j];
                //rhotot += points[j];
            }
            finufft_execute(plan2, &rho[0], &c[0]);
            
            for (size_t i = 0; i < rho.size(); i++)
            {
                Skdiff = std::real(std::abs(rho[i]));
             
                Skdiff = Skdiff*Skdiff/rhotot-Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
             
                Skdiff2 = Skdiff*Skdiff;
                phi += V[i]*Skdiff2;
                f[i] = 4/rhotot*V[i]*Skdiff*rho[i];
                if(Sk0[i] != 0)
                {
                    f[i] /= Sk0[i];
                }
            } 
            
            // Calculate Gradient
            finufft_execute(plan1, &f[0], &c[0]);
            #pragma omp parallel for
            for (size_t j = 0; j < grad.size(); j++)
            {
                grad[j]   = std::real(c[j])/(2*M_PI);
            }
            return phi;
        }
};

}
