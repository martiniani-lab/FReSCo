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

class NUwU: public BasePotential{
    //static const size_t ndim = 2;
    public:
        const size_t ndim; //dimension
        const size_t N; // Number of particles
        const std::vector<double> L; // box dimensions
        const double K; // max K magnitude to constrain
        const double beta; // Error threshhold sharpness
        const double gamma; // Error threshhold dilation
        const double eps; // finufft error tolerance
        const std::vector<int> Kvec; // Wavevectors (units of 2M_PI)
        const std::vector<double> Kmag; // Wavevector magnitudes (units of 2M_PI)
        const std::vector<double> Sk0; // Structure Factor
        const std::vector<double> V; // Potential Weighting
        const std::vector<double> radii; // Particle radii
        std::vector<std::complex<double>> c; // Point weights
        std::vector<std::complex<double>> dc; // Point weights
        const int error_mode; //Form of U(k) to be used
        const int pin_Sk; //Force continuity
        const int noisetype; // 0 for none, 1 for normal, 2 for uniform
        std::mt19937_64 generator; // rng generator
        std::normal_distribution<double> normal_distribution; // normal distribution
        std::uniform_real_distribution<double> uniform_distribution; // normal distribution
        const int radial; //Constrain only the radial S(k) (i.e, perform an angular average)
        finufft_plan plan1;
        finufft_plan plan2;
        const int periodic; //Whether to assume periodicity when calculating S(k)

        NUwU(std::vector<double> _radii, double _K, std::vector<double> _Sk, std::vector<double> _V, std::vector<double> _L, double _eps, double _beta, double _gamma, int _error_mode, int _pin_Sk, int _rseed, int _noisetype, double _stdev, int _radial, int _periodic)
        : N(_radii.size()),
          L(_L),
          ndim(_L.size()),
          K(_K),
          Kvec(calculate_Kvec(_V.size())),
          Kmag(calculate_Kmag(Kvec)),
          Sk0(_Sk),
          V(_V),
          radii(_radii),
          c(initialize_c(_radii)),
          dc(N*ndim),
          eps(_eps),
          beta(_beta),
          gamma(_gamma),
          error_mode(_error_mode),
          pin_Sk(_pin_Sk),
          noisetype(_noisetype),
          generator(size_t(_rseed)),
          normal_distribution(0.0, _stdev),
          uniform_distribution(-_stdev,_stdev),
          radial(_radial),
          periodic(_periodic)
        {
            std::vector<int64_t> nmodes(ndim,Kvec.size());
            finufft_makeplan(1, ndim, &nmodes[0], +1, 1, eps, &plan1, NULL);
            finufft_makeplan(2, ndim, &nmodes[0], -1, 1, eps, &plan2, NULL);
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

        void update_c(const std::vector<double> points)
        {
	    //c.assign(c.size(),1);
	    //dc.assign(dc.size(),0);
            std::complex<double> sum(0,0);
            std::complex<double> count(c.size(),0);
            double r = 0;
            for (size_t i=0; i < c.size(); i++)
	    {
                r = 0;
                for (size_t j=0; j<ndim; j++)
		{
                    r += (points[ndim*i+j]-0.5*L[j])*(points[ndim*i+j]-0.5*L[j]);
                }
                c[i] = radii[i]*(0.54+0.46*cos(sqrt(r/ndim)*2*M_PI));
                for (size_t j=0; j<ndim; j++)
		{
                    dc[ndim*i+j] = -radii[i]*0.46*sin(sqrt(r/ndim)*2*M_PI)*(points[ndim*i+j]-0.5*L[j])*2*M_PI/sqrt(ndim*r);
                }
                sum += c[i];
	    } 
            /*for (size_t i=0; i < c.size(); i++)
	    {
                c[i] *= count/sum;
                for (size_t j=0; j<ndim; j++)
		{
                    dc[ndim*i+j] *= count/sum*(1.0-c[i]/count);
                }
	    }*/
            return;
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
            std::vector<double> x(N); // x coordinate
	    std::vector<std::complex<double>> rho(Kmag.size());
            std::vector<double> y, z; // x coordinate
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
            double Skdiff, Skdiff2, U, dU, Skref, rhoref, Sk0ref, noise;
            double phi = 0.0;
	    int Nk = Kvec.size();
            if (periodic == 0){
                update_c(points);
            }
            
            finufft_setpts(plan1, N, &x[0], &y[0], &z[0], 0, &x[0], &y[0], &z[0]);
            finufft_execute(plan1, &c[0], &rho[0]);
            
	    if (pin_Sk > 0)
            {
                Skref = 0;
                Sk0ref = 0;
                #pragma omp parallel for
                for (size_t i = 0; i < rho.size(); i++)
                {
                    if (Kmag[i]>K && Kmag[i]<K+10)
                    {
	                rhoref = std::real(std::abs(rho[i]));
                        Skref += rhoref*rhoref/N;
                        Sk0ref += Sk0[i];
                    }
                }
                Skref /= Sk0ref;
            }
            else
                Skref = 1.0;
            for (size_t i = 0; i < rho.size(); i++)
	    {
                if (noisetype == 1)
                    noise = normal_distribution(generator)*Kmag[i]/K;
                else if (noisetype == 2)
                    noise = uniform_distribution(generator);
                else
                    noise = 0;
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/N-(1+noise)*Sk0[i]*Skref;
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
            std::vector<double> x(N); // x coordinate
	    std::vector<std::complex<double>> rho(Kmag.size()), fx(Kmag.size()), cx(N),cy(N), factor(Kmag.size());
            std::vector<double> y, z; // x coordinate
	    std::vector<std::complex<double>> fy, fz;
            #pragma omp parallel for
	    for (size_t j = 0; j < N; j++)
            {
	        x[j] = (points[ndim*j]-round(points[ndim*j]/L[0]))*2*M_PI/L[0];
            }
            if (ndim > 1)
            {
                y = std::vector<double>(N); // y coordinate
	        fy = std::vector<std::complex<double>>(Kmag.size());
                #pragma omp parallel for
	        for (size_t j = 0; j < N; j++)
	        {
                    y[j] = (points[ndim*j+1]-round(points[ndim*j+1]/L[1]))*2*M_PI/L[1];
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
	        fz = std::vector<std::complex<double>>(Kmag.size());
                #pragma omp parallel for
	        for (size_t j = 0; j < N; j++)
	        {
                    z[j] = (points[ndim*j+2]-round(points[ndim*j+2]/L[2]))*2*M_PI/L[2];
                }
            }
            else
            {
                z = std::vector<double>(1);
	        fz = std::vector<std::complex<double>>(1);
            }
            double Skdiff, Skdiff2, U, dU, Skref, rhoref, Sk0ref, noise;
            double phi = 0.0;
	    std::complex<double> Ifactor(0.0,-4.0/N);
	    grad.assign(grad.size(),0);
	    int Nk = Kvec.size();
            if (periodic == 0){
                update_c(points);
            }
            finufft_setpts(plan1, N, &x[0], &y[0], &z[0], 0, &x[0], &y[0], &z[0]);
            finufft_setpts(plan2, N, &x[0], &y[0], &z[0], 0, &x[0], &y[0], &z[0]);
            finufft_execute(plan1, &c[0], &rho[0]);
            
	    if (pin_Sk > 0)
            {
                Skref = 0;
                Sk0ref = 0;
                #pragma omp parallel for
                for (size_t i = 0; i < rho.size(); i++)
                {
                    if (Kmag[i]>K && Kmag[i]<K+10)
                    {
	                rhoref = std::real(std::abs(rho[i]));
                        Skref += rhoref*rhoref/N;
                        Sk0ref += Sk0[i];
                    }
                }
                Skref /= Sk0ref;
                //std::cout << Skref << '\n';
            }
            else
                Skref = 1.0;
            for (size_t i = 0; i < rho.size(); i++)
	    {
                if (noisetype == 1)
                    noise = normal_distribution(generator)*Kmag[i]/K;
                else if (noisetype == 2)
                    noise = uniform_distribution(generator);
                else
                    noise = 0;
	        Skdiff = std::real(std::abs(rho[i]));
             
	        Skdiff = Skdiff*Skdiff/N-(1+noise)*Sk0[i]*Skref;
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i]*Skref;
                }
             
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
	        phi += V[i]*U*Skdiff2;
	        factor[i] = (4.0/N)*V[i]*Skdiff*rho[i]*(U+Skdiff2*dU);
	        //if(Sk0[i] != 0)
                //{
                //    factor[i] /= Sk0[i]*Skref;
                //}
	        fx[i] = std::complex<double>(Kvec[int(i%Nk)])*factor[i]*-I;
                if (ndim==2)
                {
                    fy[i] = std::complex<double>(Kvec[int(i/Nk)])*factor[i]*-I;
                }
                else if (ndim == 3)
                {
		    fy[i] = std::complex<double>(Kvec[int((i%(Nk*Nk))/Nk)])*factor[i]*-I;
                    fz[i] = std::complex<double>(Kvec[int(i/(Nk*Nk))])*factor[i]*-I;
		}
	    } 
            
         
            // Calculate Gradient
            finufft_execute(plan2, &cx[0], &fx[0]);
            finufft_execute(plan2, &cy[0], &factor[0]);
            phi = 0.0;
            //#pragma omp parallel for
            for (size_t j = 0; j < N; j++)
            {
                phi += std::real(c[j]);
                grad[ndim*j]   = std::real(cx[j]*std::conj(c[j])+cy[j]*std::conj(dc[ndim*j]));
            }
            if (ndim > 1)
            {
                finufft_execute(plan2, &cx[0], &fy[0]);
                #pragma omp parallel for
                for (size_t j = 0; j < N; j++)
                {
                    grad[ndim*j+1] = std::real(cx[j]*std::conj(c[j])+cy[j]*std::conj(dc[ndim*j+1]));
                }
            }
            if (ndim > 2)
            {
                finufft_execute(plan2, &cx[0], &fz[0]);
                #pragma omp parallel for
                for (size_t j = 0; j < N; j++)
                {
                    grad[ndim*j+2] = std::real(cx[j]*std::conj(c[j])+cy[j]*std::conj(dc[ndim*j+2]));
                }
            }
            for (size_t j=0; j < grad.size(); j++)
            {
                grad[j] = std::real(dc[j]);
            }
            return phi;
        }
};

}
