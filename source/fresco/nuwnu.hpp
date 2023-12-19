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

class NUwNU: public BasePotential{
    //static const size_t ndim = 2;
    public:
        const size_t N; // Number of particles
        const std::vector<double> L; // box dimensions
        const size_t ndim; //dimension
        const size_t Nk; // Number of k vectors
        std::vector<double> kx; // K vectors to constrain
        std::vector<double> ky; // K vectors to constrain
        std::vector<double> kz; // K vectors to constrain
        const double eps; // finufft error tolerance
        const std::vector<int> Kvec; // Wavevectors (units of 2M_PI)
        const std::vector<double> Kmag; // Wavevector magnitudes (units of 2M_PI)
        const std::vector<double> Sk0; // Structure Factor
        const std::vector<double> V; // Potential Weighting
        std::vector<std::complex<double>> c; // Point weights
        finufft_plan plan3;
        finufft_plan plan3g;

        NUwNU(std::vector<double> radii, std::vector<double> _K, std::vector<double> _Sk, std::vector<double> _V, std::vector<double> _L, double _eps)
        : N(radii.size()),
          L(_L),
          ndim(_L.size()),
          Nk(size_t(_K.size()/ndim)),
          kx(Nk),
          eps(_eps),
          Sk0(_Sk),
          V(_V),
          c(initialize_c(radii))
        {
	    for (size_t j = 0; j < Nk; j++)
            {
	        kx[j] = _K[ndim*j];
            }
            if (ndim > 1)
            {
                ky = std::vector<double>(Nk); // y coordinate
                #pragma omp parallel for
	        for (size_t j = 0; j < Nk; j++)
	        {
                    ky[j] = _K[ndim*j+1];
                }
            }
            else
            {
                ky = std::vector<double>(1);
            }
            if (ndim > 2)
            {
                kz = std::vector<double>(Nk); // y coordinate
                #pragma omp parallel for
	        for (size_t j = 0; j < Nk; j++)
	        {
                    kz[j] = _K[ndim*j+2];
                }
            }
            else
            {
                kz = std::vector<double>(1);
            }
            finufft_makeplan(3, ndim, NULL, +1, 1, eps, &plan3, NULL);
            finufft_makeplan(3, ndim, NULL, -1, 1, eps, &plan3g, NULL);
        }

        std::vector<std::complex<double>> initialize_c(std::vector<double> radii)
        {
            std::vector<std::complex<double>> _c(radii.size(),1);
            double sum = 0;
            for (size_t i=0; i < _c.size(); i++)
	    {
                for (size_t j=0; j<ndim; j++)
		{
                    if(ndim ==2)
                    {
                        _c[i] = _c[i]*M_PI*radii[i]*radii[i];
                    }
                    else if(ndim == 3)
                    {
                        _c[i] = _c[i]*M_PI*radii[i]*radii[i]*radii[i]*4.0/3.0;
                    }
                }
                sum += std::real(_c[i]);
	    } 
            for (size_t i=0; i < _c.size(); i++)
	    {
                _c[i] *= _c.size()/sum;
	    }
            return _c;
        }

        void update_c(std::vector<double> radii)
        {
	    c.assign(c.size(),1);
            double sum = 0;
            for (size_t i=0; i < c.size(); i++)
	    {
                for (size_t j=0; j<ndim; j++)
		{
                    c[i] = c[i]*radii[i];
                }
                sum += std::real(c[i]);
	    } 
            for (size_t i=0; i < c.size(); i++)
	    {
                c[i] *= c.size()/sum;
	    }
            return;
        }

        virtual double get_energy(const std::vector<double>& points)
        {
            std::vector<double> x(N); // x coordinate
	    std::vector<std::complex<double>> rho(Nk);
            std::vector<double> y, z; // y,z coordinate
            #pragma omp parallel for
	    for (size_t j = 0; j < N; j++)
            {
	        x[j] = (points[ndim*j]-round(points[ndim*j]/L[0])+L[0])*2*M_PI/L[0];
            }
            if (ndim > 1)
            {
                y = std::vector<double>(N); // y coordinate
                #pragma omp parallel for
	        for (size_t j = 0; j < N; j++)
	        {
                    y[j] = (points[ndim*j+1]-round(points[ndim*j+1]/L[1])+L[1])*2*M_PI/L[1];
                }
            }
            else
            {
                y = std::vector<double>(1);
            }
            if (ndim > 2)
            {
                z = std::vector<double>(N); // y coordinate
                #pragma omp parallel for
	        for (size_t j = 0; j < N; j++)
	        {
                    z[j] = (points[ndim*j+2]-round(points[ndim*j+2]/L[2])+L[2])*2*M_PI/L[2];
                }
            }
            else
            {
                z = std::vector<double>(1);
            }
            double Skdiff, Skdiff2;
            double phi = 0.0;
            
            finufft_setpts(plan3, N, &x[0], &y[0], &z[0], Nk, &kx[0], &ky[0], &kz[0]);
            finufft_execute(plan3, &c[0], &rho[0]);
            
            for (size_t i = 0; i < rho.size(); i++)
	    {
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/N-Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
             
                Skdiff2 = Skdiff*Skdiff;
	        phi += V[i]*Skdiff2;
	    } 
            return phi;
        }

        double get_energy_gradient(const std::vector<double>& points, std::vector<double>& grad)
        {
            std::vector<double> x(N); // x coordinate
	    std::vector<std::complex<double>> rho(Nk), fx(Nk), cx(N);
            std::vector<double> y, z; // y,z coordinate
	    std::vector<std::complex<double>> fy, fz;
            #pragma omp parallel for
	    for (size_t j = 0; j < N; j++)
            {
	        x[j] = (points[ndim*j]-round(points[ndim*j]/L[0])+L[0])*2*M_PI/L[0];
            }
            if (ndim > 1)
            {
                y = std::vector<double>(N); // y coordinate
	        fy = std::vector<std::complex<double>>(Nk);
                #pragma omp parallel for
	        for (size_t j = 0; j < N; j++)
	        {
                    y[j] = (points[ndim*j+1]-round(points[ndim*j+1]/L[1])+L[1])*2*M_PI/L[1];
                }
            }
            else
            {
                y = std::vector<double>(1);
	        fy = std::vector<std::complex<double>>(1);
            }
            if (ndim > 2)
            {
                z = std::vector<double>(N); // y coordinate
	        fz = std::vector<std::complex<double>>(Nk);
                #pragma omp parallel for
	        for (size_t j = 0; j < N; j++)
	        {
                    z[j] = (points[ndim*j+2]-round(points[ndim*j+2]/L[2])+L[2])*2*M_PI/L[2];
                }
            }
            else
            {
                z = std::vector<double>(1);
	        fz = std::vector<std::complex<double>>(1);
            }
            double Skdiff, Skdiff2;
	    std::complex<double> factor;
            double phi = 0.0;
	    std::complex<double> Ifactor(0.0,-4.0/N);
	    grad.assign(grad.size(),0);
            
            finufft_setpts(plan3, N, &x[0], &y[0], &z[0], Nk, &kx[0], &ky[0], &kz[0]);
            finufft_execute(plan3, &c[0], &rho[0]);
            finufft_setpts(plan3g, Nk, &kx[0], &ky[0], &kz[0], N, &x[0], &y[0], &z[0]);
            
            for (size_t i = 0; i < rho.size(); i++)
	    {
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/N-Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
             
                Skdiff2 = Skdiff*Skdiff;
	        phi += V[i]*Skdiff2;
	        factor = Ifactor*V[i]*Skdiff*rho[i];
	        if(Sk0[i] != 0)
                {
                    factor /= Sk0[i];
                }
	        fx[i] = std::complex<double>(kx[i])*factor;
                if (ndim>1)
                {
                    fy[i] = std::complex<double>(ky[i])*factor;
                }
                if (ndim >2)
                {
                    fz[i] = std::complex<double>(kz[i])*factor;
		}
	    } 
            
         
            // Calculate Gradient
            finufft_execute(plan3g, &fx[0], &cx[0]);
            #pragma omp parallel for
            for (size_t j = 0; j < N; j++)
            {
                grad[ndim*j]   = std::real(cx[j]);
            }
            if (ndim > 1)
            {
                finufft_execute(plan3g, &fy[0], &cx[0]);
                #pragma omp parallel for
                for (size_t j = 0; j < N; j++)
                {
                    grad[ndim*j+1] = std::real(cx[j]);
                }
            }
            if (ndim > 2)
            {
                finufft_execute(plan3g, &fz[0], &cx[0]);
                #pragma omp parallel for
                for (size_t j = 0; j < N; j++)
                {
                    grad[ndim*j+2] = std::real(cx[j]);
                }
            }
            return phi;
        }
};

}
