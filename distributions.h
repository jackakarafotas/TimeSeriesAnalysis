#ifndef __DISTRIBUTIONS_H
#define __DISTRIBUTIONS_H

#include <iostream>
using namespace std;

class Distributions {
public:
	// Constructor
	Distributions();

	// Virtual Destructor
	virtual ~Distributions();

	double gamma(const double& x, const double& alpha, const double& beta);
	double chi_squared(const double& x, const double& k);
	double normal(const double& x, const double& mu, const double& sigma);
	double inv_normal(const double& perc, const double& mu, const double& sigma);
	double student_t(const double& x, const double& df);
	double beta_dist(const double& x, const double& alpha, const double& beta);
	double inverse_gamma(const double& x, const double& kappa, const double& inv_precision);
};


#endif