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
        const size_t ndim; //dimension
        const std::vector<double> L; // box dimensions
        const std::vector<int> N; // grid size
        const double K; // Max K magnitude
        const double beta; // Error threshhold sharpness
        const double gamma; // Error threshhold dilation
        const double eps; // finufft error tolerance
        const std::vector<int> Kvec; // Wavevectors (units of 2M_PI)
        const std::vector<double> Kmag; // Wavevector magnitudes (units of 2M_PI)
        const std::vector<double> Sk0; // Structure Factor
        const std::vector<double> V; // Potential Weighting
        std::complex<double>* c; // Point weights
        std::complex<double>* rho; // Point weights
        std::complex<double>* f; // Point weights
        const int error_mode; //Form of U(k) to be used
        const int pin_Sk; //Force continuity
        const int noisetype; // 0 for none, 1 for normal, 2 for uniform
        std::mt19937_64 generator; // rng generator
        std::normal_distribution<double> normal_distribution; // normal distribution
        std::uniform_real_distribution<double> uniform_distribution; // normal distribution
        fftw_plan plan1;
        fftw_plan plan2;

        UwU(std::vector<int> _N, double _K, std::vector<double> _Sk, std::vector<double> _V, std::vector<double> _L, double _eps, double _beta, double _gamma, int _error_mode, int _pin_Sk, int _rseed, int _noisetype, double _stdev)
        : L(_L),
          N(_N),
          ndim(_L.size()),
          K(_K),
          Kvec(calculate_Kvec(_V.size())),
          Kmag(calculate_Kmag(Kvec)),
          Sk0(_Sk),
          V(_V),
          eps(_eps),
          beta(_beta),
          gamma(_gamma),
          error_mode(_error_mode),
          pin_Sk(_pin_Sk),
          noisetype(_noisetype),
          generator(size_t(_rseed)),
          normal_distribution(0.0, _stdev),
          uniform_distribution(-_stdev,_stdev)
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

        void calculate_U(double& U, double& dU, double Skdiff2, double Kval)
        {
            if(error_mode == 0)
            {
                U = 1;
                dU = 0;
                return;
            }
            U = 1/(1+exp(beta*(gamma*(1+Kval/K)-Skdiff2)));
            dU = beta*U*(1-U);
            if(error_mode == 2 && U < 0.2)
            {
                U = 0;
                dU = 0;
            }
            return;
        }
        virtual double get_energy(const std::vector<double>& points)
        {
            double Skdiff, Skdiff2, U, dU, Skref, rhoref, Sk0ref, noise;
            double phi = 0.0;
            double rhotot = 0.0;
            #pragma omp parallel for
            for (size_t j = 0; j < points.size(); j++)
            {
                c[j] = points[j];
                rhotot += points[j];
            }
            fftw_execute(plan1);
            
	    if (pin_Sk > 0)
            {
                Skref = 0;
                Sk0ref = 0;
                #pragma omp parallel for
                for (size_t i = 0; i < points.size(); i++)
                {
                    if (Kmag[i]>K && Kmag[i]<K+10)
                    {
	                rhoref = std::real(std::abs(rho[i]));
                        Skref += rhoref*rhoref/rhotot;
                        Sk0ref += Sk0[i];
                    }
                }
                Skref /= Sk0ref;
                //std::cout << Skref << '\n';
            }
            else
                Skref = 1.0;
            for (size_t i = 0; i < points.size(); i++)
	    {
                if (noisetype == 1)
                    noise = normal_distribution(generator)*Kmag[i]/K;
                else if (noisetype == 2)
                    noise = uniform_distribution(generator);
                else
                    noise = 0;
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/rhotot-(1+noise)*Sk0[i]*Skref;
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i]*Skref;
                }
             
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
	        phi += V[i]*U*Skdiff2;
	    } 
            return phi;
        }

        virtual double get_energy_gradient(const std::vector<double>& points, std::vector<double>& grad)
        {
            double Skdiff, Skdiff2, U, dU, Skref, rhoref, Sk0ref, noise;
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
            
	    if (pin_Sk > 0)
            {
                Skref = 0;
                Sk0ref = 0;
                #pragma omp parallel for
                for (size_t i = 0; i < points.size(); i++)
                {
                    if (Kmag[i]>K && Kmag[i]<K+10)
                    {
	                rhoref = std::real(std::abs(rho[i]));
                        Skref += rhoref*rhoref/rhotot;
                        Sk0ref += Sk0[i];
                    }
                }
                Skref /= Sk0ref;
                //std::cout << Skref << '\n';
            }
            else
                Skref = 1.0;
            for (size_t i = 0; i < points.size(); i++)
	    {
                if (noisetype == 1)
                    noise = normal_distribution(generator)*Kmag[i]/K;
                else if (noisetype == 2)
                    noise = uniform_distribution(generator);
                else
                    noise = 0;
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/rhotot-(1+noise)*Sk0[i]*Skref;
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i]*Skref;
                }
             
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
	        phi += V[i]*U*Skdiff2;
	        f[i] = 2/rhotot*V[i]*Skdiff*rho[i]*(U+Skdiff2*dU);
	        if(Sk0[i] != 0)
                {
                    f[i] /= Sk0[i]*Skref;
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
