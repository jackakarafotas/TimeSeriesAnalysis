#ifndef __BETA_CPP
#define __BETA_CPP

#include "beta.h"
#include "arfima.cpp"
#include "autocorrelation.cpp"

// Main
Beta::Beta(VectorXf& ts1, VectorXf ts2) {
	if (ts1.size() != ts2.size())
		throw invalid_argument( "Size of two time series must be equal" );
	time_series_1 = ts1; 
	time_series_2 = ts2;
}
Beta::~Beta() {}


// TRAIN
void Beta::_train_recur(
	VectorXf& ts_x,
	VectorXf& ts_y,
	VectorXf& resid,
	VectorXf& coeffs,
	int& _d) {
	// Train regression till residuals are not random walk
	MatrixXf A = transform_X(ts_x);

	// 1. fit linear regression model and check serial corr of residuals
	regression(A,ts_y,resid,coeffs);

	// 2. check if residuals are unit root stationary
	ARFIMA model(resid);
	bool unit_root_nonstationary = model.unit_root_test(resid);
	if (unit_root_nonstationary) {
		_d++;
		VectorXf ts_x = model.difference(ts_x, 1);
		VectorXf ts_y = model.difference(ts_y, 1);
		_train_recur(
			ts_x, 
			ts_y, 
			resid, 
			coeffs,
			_d);
	}
}

void Beta::train(const bool& predict_ts1) {
	// Train Linear model between time series
	VectorXf dep_ts;
	VectorXf indep_ts;
	if (predict_ts1) {
		dep_ts = time_series_1;
		indep_ts = time_series_2;
	} else {
		indep_ts = time_series_1;
		dep_ts = time_series_2;
	}

	// 1. + 2. keep fitting till residuals aren't random walk
	_train_recur(
		dep_ts,
		indep_ts,
		residuals,
		coefficients,
		d);

	// 3. Train model
	MatrixXf A(dep_ts.size()-1,3);
	VectorXf b(dep_ts.size()-1);

	for (int i = 1; i < dep_ts.size(); i++) {
		A(i-1,0) = 1;
		A(i-1,1) = indep_ts(i);
		A(i-1,2) = residuals(i-1); // AR on the residuals

		b(i-1) = dep_ts(i);
	}

	regression(A,b,residuals,coefficients);
	beta = coefficients(1);
}

void Beta::regression(
	const MatrixXf& X,
	const VectorXf& y,
	VectorXf& resid,
	VectorXf& coeffs) {
	coeffs = X.jacobiSvd(ComputeThinU | ComputeThinV).solve(y);
	resid = y - X*coeffs;
}

MatrixXf Beta::transform_X(const VectorXf& ts) const {
	MatrixXf X(ts.size(),2);

	for (int i = 0; i < ts.size(); i++) {
		X(i,0) = 1;
		X(i,1) = ts(i);
	}

	return X;
}


// TEST
bool Beta::test_residual_autocorrelation(const int& lag) const {
	// Tests residual autocorrelation
	AutoCorrelation ac;
	return ac.test_autocorrelation(residuals, coefficients.size()-1+lag, lag);
}


// PREDICT
VectorXf Beta::predict(const VectorXf& indep_ts, const VectorXf& dep_ts) {
	// GET RID OF DEPENDENT TIME SERIES
	if (indep_ts.size() != dep_ts.size())
		throw invalid_argument( "Size of two time series must be equal" );

	ARFIMA model(indep_ts);
	VectorXf indep = model.difference(indep_ts,d);
	VectorXf dep = model.difference(dep_ts,d);
	VectorXf resid;
	VectorXf coeffs;

	// Get residuals
	MatrixXf X = transform_X(indep);
	regression(X,dep,resid,coeffs);

	// Add AR(1) on the residuals
	MatrixXf A(indep.size()-1,3);
	VectorXf b(indep.size()-1);

	for (int i = 1; i < indep.size(); i++) {
		A(i-1,0) = 1;
		A(i-1,1) = indep(i);
		A(i-1,2) = resid(i-1); // AR on the residuals

		b(i-1) = dep(i);
	}

	// Adjust for U
	return data * coefficients;
}


#endif