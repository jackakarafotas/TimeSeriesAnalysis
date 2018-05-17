#ifndef __BETA_H
#define __BETA_H

#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
using namespace std;
using namespace Eigen;

class Beta {
private:
	VectorXf time_series_1;
	VectorXf time_series_2;
	VectorXf residuals;
	VectorXf coefficients;
	int d;

public:
	double beta;
	vector<bool> significant;

	// Constructor
	Beta(VectorXf& ts1, VectorXf ts2);

	// Virtual Destructor
	virtual ~Beta();

	// Train
	void train(const bool& predict_ts1 = true);
	void regression(
		const MatrixXf& X,
		const VectorXf& y,
		VectorXf& resid,
		VectorXf& coeffs);
	void _train_recur(
		VectorXf& ts_x,
		VectorXf& ts_y,
		VectorXf& resid,
		VectorXf& coeffs,
		int& _d);

	MatrixXf transform_X(const VectorXf& ts) const;

	// Test
	bool test_residual_autocorrelation(const int& lag) const;

	// Predict
	VectorXf predict(const VectorXf& ts1, const VectorXf& ts2);

	
};


#endif