#ifndef __DISTRIBUTIONS_CPP
#define __DISTRIBUTIONS_CPP

#include <iostream>
#include <cmath>
#include <tgmath.h>
#include <math.h>
#include "distributions.h"
using namespace std;

// Helper
float erfinv(float x){
   float tt1, tt2, lnx, sgn;
   sgn = (x < 0) ? -1.0f : 1.0f;

   x = (1 - x)*(1 + x);        // x = 1 - x*x;
   lnx = logf(x);

   tt1 = 2/(M_PI*0.147) + 0.5f * lnx;
   tt2 = 1/(0.147) * lnx;

   return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}

// Main
Distributions::Distributions() {}

Distributions::~Distributions() {}

double Distributions::gamma(const double& x, const double& alpha, const double& beta) {
	long double probability = 1/tgammal(alpha);
	probability *= pow(beta,alpha);
	probability *= pow(x,alpha-1);
	probability *= exp(-beta*x);

	return probability;
}

double Distributions::chi_squared(const double& x, const double& k) {
	return gamma(x,k/2.0,0.5);
}

double Distributions::normal(const double& x, const double& mu, const double& sigma) {
	double probability = 1/pow(2*M_PI,0.5);
	probability /= sigma;
	probability *= exp(-pow(x-mu,2.0)/(2*pow(sigma,2)));
	return probability;
}

double Distributions::inv_normal(const double& perc, const double& mu, const double& sigma) {
	return mu + (sigma * pow(2,0.5) * erfinv((2*perc) - 1));
}

double Distributions::student_t(const double& x, const double& df) {
	if (df < 50.0) {
		// If low Degrees of freedom do student t
		long double probability = tgammal((df+1.0)/2.0);
		probability /= (pow(df*M_PI,0.5) * tgammal(df/2.0));
		probability *= pow(1.0 + (pow(x,2.0)/df),-(df+1.0)/2.0);
		return probability;
	} else {
		// If high degrees of freedom, estimate using a standard normal
		return normal(x,0,1);
	}
}


#endif