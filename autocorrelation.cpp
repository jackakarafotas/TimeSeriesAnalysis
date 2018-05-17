#ifndef __AUTOCORRELATION_CPP
#define __AUTOCORRELATION_CPP

#include <iostream>
#include <math.h>
#include <cmath>
#include "autocorrelation.h"
#include "distributions.cpp"
using namespace std;

// INIT

AutoCorrelation::AutoCorrelation() {}
AutoCorrelation::~AutoCorrelation() {}

double AutoCorrelation::autocorrelation(
	const VectorXf& returns,
	const int& lag) const {
	double cov = 0;
	double var = 0;
	double return_mean = returns.mean();

	// Covariance
	for (int t = lag; t < returns.size(); t++) {
		cov += (returns(t) - return_mean) * (returns(t-lag) - return_mean);
	}

	// Variance
	for (int t = 0; t < returns.size(); t++) {
		var += pow(returns(t) - return_mean,2.0);
	}

	return cov / var;
}

bool AutoCorrelation::test_autocorrelation(
	const VectorXf& returns,
	const int& max_lag,
	const int& degrees_freedom,
	const double &alpha) const {
	// Ljung Box test, to test several autocorrelations

	// Create test statistic
	double ljung_box = 0;
	for (int l = 1; l <= max_lag; l++) {
		ljung_box += pow(autocorrelation(returns, l),2.0) / (returns.size() - l);
	}
	ljung_box *= returns.size() * (returns.size() + 2);

	// Calculate p value
	Distributions pdf;
	double p_value = pdf.chi_squared(ljung_box, degrees_freedom);

	// Accept or reject
	if (p_value <= alpha) {
		return true; // Null Hypothesis is rejected
	} else {
		return false; // Null Hypothesis is accepted
	}
}

bool AutoCorrelation::test_ac_lag(
	const VectorXf& returns,
	const int& lag,
	const double& alpha) const {
	// Tests whether an autocorrelation at a specific lag exists
	// True = null hypothsis is rejected (null = does not exist)
	double sum = 0;

	// Calc statistic
	for (int l = 1; l < lag; l++)
		sum += pow(autocorrelation(returns,l),2.0);

	double t_ratio = autocorrelation(returns,lag);
	t_ratio /= pow((1 + (2*sum)) / returns.size(),0.5);
	
	// Calc p value and accept or reject
	Distributions pdf;
	double p_value = pdf.normal(t_ratio, 0.0, 1.0);
	if (p_value <= (alpha/2.0)) {
		return true; 
	} else {
		return false;
	}
}


#endif