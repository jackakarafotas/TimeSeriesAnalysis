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

double Distributions::beta_dist(const double& x, const double& alpha, const double& beta) {
	if ((x < 0) || (x > 1)) 
		throw invalid_argument( "x must be between 0 and 1." );
	if ((alpha < 0) || (beta < 0))
		throw invalid_argument( "alpha and beta must be greater than 0." );

	double probability = pow(x, alpha-1) * pow(1-x,beta-1);
	probability /= ((tgammal(alpha) * tgammal(beta)) / tgammal(alpha+beta));
	return probability;
}

double Distributions::inverse_gamma(const double& x, const double& kappa, const double& inv_precision) {
	double probability = pow(kappa * pow(inv_precision,2.0), kappa/2.0);
	probability /= (pow(2.0, kappa/2.0) * tgammal(kappa/2.0));
	probability *= pow(1/pow(x,2.0),(kappa/2.0) + 1.0);
	probability *= exp(-0.5 * kappa * pow(inv_precision, 2.0) / pow(x, 2.0));
	return probability;
}


#endif