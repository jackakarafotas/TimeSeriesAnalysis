#ifndef __ARFIMA_H
#define __ARFIMA_H

#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
using namespace std;
using namespace Eigen;

class ARFIMA {
	// workflow:
	// -> set up ARIMA object
	// -> difference the time series
	// -> set the time series
	// -> train test split
	// -> train the model
	// -> validate the model
private:
	int p; 						// number of AR coefficients
	int q;						// number of MA coefficients
	int u_p; 					// number of AR coefficients in order to estimate AR residuals (to be used by the ARMA model)
	VectorXf u;					// AR residuals (to be used in the ARMA model)

	VectorXf time_series;		// input time series
	MatrixXf A;					// Matrix of inputs to regress on
	VectorXf b;					// Output labels for regression
	VectorXf differencing_weights;// weights used to difference a time series
	vector<bool> significant;	// Whether coefficients are significant

	double sd_resid;			// standard deviation of the residuals 
	double sse;					// sum of squared errors

public:
	bool trained;				// whether the model is trained or not
	VectorXf coefficients;		// the model coefficients
	VectorXf residuals;			// the model residuals
	VectorXf training;			// data to train on
	VectorXf validation;		// data to validate on
	VectorXf validation_pred;	// predictions on the validation set
	VectorXf validation_comp;	// to compare the predictions with

	int optimal_p;				// the optimal p value found by pick_parameters
	float optimal_d;			// the optimal d value found by pick_d
	int optimal_q;				// the optimal q value found by pick_parameters
	int k_weights;				// Number of weights used to difference the time series
	int adj_size;				// how much to adjust the size of the time series by (since ARMA crops a time series)
	double r2;					// the R2 of the model
	double r2_adj;				// the adjusted R2 of the model
	double aic;					// Aikake Information Criterion of the model

	/* DEFAULT */
	// Constructor
	ARFIMA();
	ARFIMA(const VectorXf& _time_series);
	// Virtual Destructor
	virtual ~ARFIMA();
	void set_time_series(const VectorXf& _time_series);


	// Transform data
	void train_test_split(const double& ratio = 0.75);
	MatrixXf transform_X(const VectorXf& _time_series,const VectorXf& _u) const;
	VectorXf transform_y(const VectorXf& _time_series,const VectorXf& _u) const;


	/* TRAIN PREDICT */
	// Train
	void train(const int& _p, const int& _q); 
	VectorXf _autoregression(
		const VectorXf& _time_series,
		const int& _p,
		VectorXf& _coefficients) const;

	// Predict
	void validate();
	VectorXf predict(const VectorXf& ts) const;
	void predict_bounds(
		const VectorXf& prediction,
		const double& range_perc,
		VectorXf& upper,
		VectorXf& lower) const;


	/* TESTING */
	// Test regression
	double AIC(const int& _p, const int& _q);
	vector<bool> test_coefficient_significance(const double& alpha = 0.05);
	MatrixXf coefficient_covariance();
	double R2() const;
	double adjusted_R2() const;
	double MSE(const VectorXf& prediction,const VectorXf& actual) const;

	// Test Autocorrelation
	bool test_autocorrelation(const VectorXf& _time_series,const int& lag) const;
	bool test_residual_autocorrelation(const int& lag) const;
	bool test_arch_effect(
		const int& m,
		const double& alpha = 0.05);


	/* PICK PARAMS */
	// optimize p,q,d
	tuple<int, int> pick_parameters(const int& max_p, const int& max_q);
	float pick_d(const VectorXf& _time_series,const float& max_d);


	// Differencing and LT Memory
	bool unit_root_test(
		const VectorXf& _time_series,
		const double& alpha = 0.05) const;

	VectorXf difference(
		const VectorXf& _time_series,
		const float& d);

	VectorXf undifference_prediction(
		const VectorXf& orig_prices,
		const VectorXf& differenced_predictions,
		const float& d);

	VectorXf _get_weights(
		const int& time_series_size,
		const float& d,
		const float& weight_cutoff = 1e-5,
		const int& max_weights_vector_size = 300);

	double rescaled_range(const int& start, const int& end) const;
	
};


#endif