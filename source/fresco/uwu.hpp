#include <string>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "base_potential.hpp"
#include <vector>
#include <complex>
#include <random>
#include <fftw3.h>

//static const double M_PI = 3.1415926;
static const std::complex<double> I(0.0,1.0);

namespace fresco{

class UwU: public BasePotential{
    public:
        const std::vector<double> L; // box dimensions
        const std::vector<int> N; // grid size
        const size_t ndim; //dimension
        const double K; // Max K magnitude
        const double eps; // finufft error tolerance
        const std::vector<int> Kvec; // Wavevectors (units of 2M_PI)
        const std::vector<double> Kmag; // Wavevector magnitudes (units of 2M_PI)
        const std::vector<double> Sk0; // Structure Factor
        const std::vector<double> V; // Potential Weighting
        std::complex<double>* c; // Point weights
        std::complex<double>* rho; // Point weights
        std::complex<double>* f; // Point weights
        fftw_plan plan1;
        fftw_plan plan2;

        UwU(std::vector<int> _N, double _K, std::vector<double> _Sk, std::vector<double> _V, std::vector<double> _L, double _eps)
        : L(_L),
          N(_N),
          ndim(_L.size()),
          K(_K),
          eps(_eps),
          Kvec(calculate_Kvec(_V.size())),
          Kmag(calculate_Kmag(Kvec)),
          Sk0(_Sk),
          V(_V)
        {
          c = (std::complex<double>*) fftw_malloc(_Sk.size()*sizeof(std::complex<double>));
          rho = (std::complex<double>*) fftw_malloc(_Sk.size()*sizeof(std::complex<double>));
          f = (std::complex<double>*) fftw_malloc(_Sk.size()*sizeof(std::complex<double>));
          plan1 = fftw_plan_dft(ndim, &N[0], reinterpret_cast<fftw_complex*>(c), reinterpret_cast<fftw_complex*>(rho), +1, FFTW_MEASURE);
          plan2 = fftw_plan_dft(ndim, &N[0], reinterpret_cast<fftw_complex*>(f), reinterpret_cast<fftw_complex*>(c), -1, FFTW_MEASURE);
        }

        std::vector<int> calculate_Kvec(size_t Vsize)
        {
            size_t K_;
            if (ndim ==2)
            {
                K_ = size_t((sqrt(Vsize)-1)/2);
            }
            else if (ndim == 3)
            {
                K_ = size_t((cbrt(Vsize)-1)/2);
            }
            else
            {
                 throw std::runtime_error("Invalid dimension: "+std::to_string(ndim));
            }
            std::vector<int> _Kvec(K_*2+1);
            for (size_t i=0; i < _Kvec.size(); i++)
	    {
		_Kvec[i] = int(i)-int(K_);
	    }
            return _Kvec;
        }

        std::vector<double> calculate_Kmag(std::vector<int> Kvec)
        {
            if(ndim == 2)
            {
                std::vector<double> _Kmag(Kvec.size()*Kvec.size());
                for (size_t i=0; i < Kvec.size(); i++)
   	        {
                    for(size_t j=0; j < Kvec.size(); j++)
		    {
                        _Kmag[i+Kvec.size()*j] = sqrt(Kvec[i]*Kvec[i]+Kvec[j]*Kvec[j]);
                    }
	        }
                return _Kmag;
            }
            else if(ndim == 3)
            {
                std::vector<double> _Kmag(Kvec.size()*Kvec.size()*Kvec.size());
                for (size_t i=0; i < Kvec.size(); i++)
   	        {
                    for(size_t j=0; j < Kvec.size(); j++)
		    {     
                        for(size_t k=0; k < Kvec.size(); k++)
                        {
                            _Kmag[i+Kvec.size()*j+Kvec.size()*Kvec.size()*k] = sqrt(Kvec[i]*Kvec[i]+Kvec[j]*Kvec[j]+Kvec[k]*Kvec[k]);
                        }
                    }
	        }
                return _Kmag;
            }
            else
            {
                 throw std::runtime_error("Invalid dimension: "+std::to_string(ndim));
            }
        }

        virtual double get_energy(const std::vector<double>& points)
        {
            double Skdiff, Skdiff2;
            double phi = 0.0;
            double rhotot = 0.0;
            #pragma omp parallel for
            for (size_t j = 0; j < points.size(); j++)
            {
                c[j] = points[j];
                rhotot += points[j];
            }
            fftw_execute(plan1);
            
            for (size_t i = 0; i < points.size(); i++)
	    {
	        Skdiff = std::real(std::abs(rho[i]));
             
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
            double Skdiff, Skdiff2;
            double phi = 0.0;
	    grad.assign(grad.size(),0);
            double rhotot = 0.0;
            #pragma omp parallel for
            for (size_t j = 0; j < points.size(); j++)
            {
                c[j] = points[j];
                rhotot += points[j];
            }
            fftw_execute(plan1);
            
            for (size_t i = 0; i < points.size(); i++)
	    {
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/rhotot-Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
             
                Skdiff2 = Skdiff*Skdiff;
	        phi += V[i]*Skdiff2;
	        f[i] = 2/rhotot*V[i]*Skdiff*rho[i];
	        if(Sk0[i] != 0)
                {
                    f[i] /= Sk0[i];
                }
	    } 
            
         
            // Calculate Gradient
            fftw_execute(plan2);
            #pragma omp parallel for
            for (size_t j = 0; j < grad.size(); j++)
            {
                grad[j]   = std::real(c[j]);
            }
            return phi;
        }
};

}
