#ifndef __AUTOCORRELATION_H
#define __AUTOCORRELATION_H

#include <iostream>
#include <eigen/Eigen/Dense>
using namespace std;
using namespace Eigen;

class AutoCorrelation {
public:

	// Constructor
	AutoCorrelation();

	// Virtual Destructor
	virtual ~AutoCorrelation();

	// Main methods
	// Autocorrelation
	double autocorrelation(
		const VectorXf& returns,
		const int& lag) const;

	bool test_autocorrelation(
		const VectorXf& returns, 
		const int& max_lag, 
		const int& degrees_freedom,
		const double& alpha = 0.05) const;

	bool test_ac_lag(
		const VectorXf& returns,
		const int& lag,
		const double& alpha = 0.05) const;
};





#endif