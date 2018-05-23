#ifndef __ARFIMA_CPP
#define __ARFIMA_CPP

#include <cmath>
#include <algorithm>
#include "arfima.h"
#include "autocorrelation.cpp"
#include "distributions.cpp"


/* MAIN */
// Default
ARFIMA::ARFIMA() {}
ARFIMA::ARFIMA(const VectorXf& _time_series) : time_series(_time_series) {}
ARFIMA::~ARFIMA() {}
void ARFIMA::set_time_series(const VectorXf& _time_series) { time_series = _time_series; }


// transform data
void ARFIMA::train_test_split(const double& ratio) {
	// Splits the training data into training and validation
	unsigned int cutoff = floor(ratio*time_series.size());
	training = time_series.head(cutoff);
	validation = time_series.tail(time_series.size()-cutoff);
}

MatrixXf ARFIMA::transform_X(const VectorXf& _time_series,const VectorXf& _u) const {
	// Transform time series data for model
	// Get the independent variable matrix

	MatrixXf X(_time_series.size()-adj_size,p+q+1);
	for (int i = adj_size; i < _time_series.size(); i++) {
		X(i-adj_size,0) = 1;

		for (int j = 1; j <= p; j++)
			X(i-adj_size,j) = _time_series(i-j);

		for (int j = 1; j <= q; j++)
			X(i-adj_size,j+p) = _u(i-u_p-j);
	}

	return X;
}

VectorXf ARFIMA::transform_y(const VectorXf& _time_series,const VectorXf& _u) const {
	// Transform time series data for model
	// Get the dependent variable vector

	VectorXf y(_time_series.size()-adj_size);
	for (int i = adj_size; i < _time_series.size(); i++) {
		y(i-adj_size) = _time_series(i);
	}

	return y;
}



/* TRAIN PREDICT */
// trainer
VectorXf ARFIMA::_autoregression(
	const VectorXf& _time_series,
	const int& _p,
	VectorXf& _coefficients) const {
	// returns residuals
	// optionally retruns coefficients

	// Get A and b matrices
	MatrixXf X(_time_series.size()-_p,_p+1);
	VectorXf y(_time_series.size()-_p);

	for (int i = _p; i < _time_series.size(); i++) {
		y(i-_p) = _time_series(i);
		X(i-_p,0) = 1;
		for (int j = 1; j <= _p; j++) {
			X(i-_p,j) = _time_series(i-j);
		}
	}

	// Solve least squares problem
	_coefficients = X.jacobiSvd(ComputeThinU | ComputeThinV).solve(y);

	// Get residuals
	return y - X*_coefficients;
}

void ARFIMA::train(const int& _p, const int& _q) {
	// Train ARFIMA using two step regression method (Hannan–Rissanen algorithm)
	p = _p;
	q = _q;

	// Regress to estimate u values for moving average model
	u_p = 20;
	VectorXf u_coeffs(u_p+1);
	u = _autoregression(training,u_p,u_coeffs);

	// Get A and b matrices
	adj_size = max(u_p+q,p);
	A = transform_X(training,u);
	b = transform_y(training,u);

	// Solve least squares problem
	coefficients = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

	// Get residuals
	VectorXf output = A*coefficients;
	residuals = b - output;

	// Calc stats
	sse = 0;
	for (int i = 0; i < residuals.size(); i++)
		sse += pow(residuals(i),2.0);
	sd_resid = pow(sse / (residuals.size() - p - q - 1),0.5);

	trained = true;
}

void ARFIMA::validate() {
	// Validate model using validation set
	VectorXf u_coeffs(u_p+1);
	VectorXf u_predict = _autoregression(validation,u_p,u_coeffs);
	MatrixXf data = transform_X(validation,u_predict);
	VectorXf v_pred = (data * coefficients);

	validation_pred = v_pred.head(v_pred.size()-1); // remove last element b/c can't compare it
	validation_comp = validation.tail(validation_pred.size());
}

VectorXf ARFIMA::predict(const VectorXf& ts) const {
	// Predict on an unseen data set
	// Assumes data is differenced already
	VectorXf u_coeffs(u_p+1);
	VectorXf u_predict = _autoregression(ts,u_p,u_coeffs);
	MatrixXf data = transform_X(ts,u_predict);
	VectorXf prediction = (data * coefficients); 
	return prediction;
}


void ARFIMA::predict_bounds(
	const VectorXf& prediction,
	const double& range_perc,
	VectorXf& upper,
	VectorXf& lower) const {
	// Upper and lower bounds for our prediction
	Distributions inv_cdf;
	double p = range_perc + ((1.0 - range_perc)/2);
	double z = inv_cdf.inv_normal(p,0,1);

	VectorXf perturb = VectorXf::Ones(upper.size())*z*sd_resid;
	upper = prediction + perturb;
	lower = prediction - perturb;
}



/* TESTING */
double ARFIMA::AIC(const int& _p, const int& _q) {
	// Aikake Information Criterion for Least Squares
	train(_p,_q);
	aic = log(pow(sd_resid,2.0)) + (2*(_p+_q)/(training.size()));
	return aic;
}

MatrixXf ARFIMA::coefficient_covariance() {
	// get the covariance for each coefficient
	MatrixXf A_centered = A;
	double feature_mean;
	for (int c = 0; c < A.cols(); c++) {
		feature_mean = A.col(c).mean();
		for (int r = 0; r < A.rows(); r++)
			A_centered(r,c) -= feature_mean;
	}
	return A_centered.transpose() * A_centered / A.rows();
}


// Test significance
vector<bool> ARFIMA::test_coefficient_significance(const double& alpha) {
	// Test the significance of each coefficient
	// Calculate T statistics
	MatrixXf cov = coefficient_covariance();
	VectorXf t_stats(coefficients.size());
	for (int i = 0; i < coefficients.size(); i++) {
		t_stats(i) = coefficients[i] * pow(A.rows() * cov(i,i), 0.5) / sd_resid;
	}

	// Calculate P values
	Distributions pdf;
	VectorXf p_values(t_stats.size());
	for (int i = 0; i < t_stats.size(); i++)
		p_values(i) = pdf.student_t(t_stats[i], A.rows()-1);

	// Get significance
	vector<bool> significant;
	for (int i = 0; i < t_stats.size(); i++) {
		if (p_values(i) <= (alpha/2.0)) {
			significant.push_back(true);
		} else {
			significant.push_back(false);
		}
	}
	return significant;
}

bool ARFIMA::test_autocorrelation(const VectorXf& _time_series,const int& lag) const {
	// Tests whether autocorrelation is significant for the stock being analyzed
	// Uses the Ljung-Box statistic
	AutoCorrelation ac;
	return ac.test_autocorrelation(_time_series,lag,lag);
}

bool ARFIMA::test_residual_autocorrelation(const int& lag) const {
	// Tests whether autocorrelation is significant for the residuals
	// Uses the Ljung-Box statistic
	AutoCorrelation ac;
	return ac.test_autocorrelation(residuals, coefficients.size()-1+lag, lag);
}

double ARFIMA::R2() const {
	// Calculates the R2 for the model
	double ts_mean = b.mean();
	double tss = 0;
	for (int i = 0; i < b.size(); i++) 
		tss += pow(b(i) - ts_mean,2.0);

	return 1 - (sse / tss);
}

double ARFIMA::adjusted_R2() const {
	// Calculates the adjusted R2 for the model
	double ts_mean = b.mean();
	double ts_var = 0;
	for (int i = 0; i < b.size(); i++) 
		ts_var += pow(b(i) - ts_mean,2.0);
	ts_var /= (b.size()-1);

	return 1 - (pow(sd_resid,2.0) / ts_var);
}

double ARFIMA::MSE(const VectorXf& prediction,const VectorXf& actual) const {
	if (prediction.size() != actual.size())
		throw invalid_argument( "Size of two time series must be equal" );

	VectorXf error = prediction - actual;

	// Calc stats
	double test_sse = 0;
	for (int i = 0; i < error.size(); i++)
		test_sse += pow(error(i),2.0);

	return test_sse / error.size();
}

bool ARFIMA::test_arch_effect(
	const int& m,
	const double& alpha) {
	// Test for arch effect after fitting ARIMA
	// True --> signficant arch effect
	// Tests whether significant autocorrelaiton in squared residuals

	// Cannot test if not trained 
	if (!trained) {
		throw invalid_argument( "Train model first." );
	}

	// Square the residuals
	VectorXf squared_residuals = residuals;
	for (int i = 0; i < squared_residuals.size(); i++)
		squared_residuals(i) = pow(squared_residuals(i),2.0);

	// Train AR on squared residuals
	VectorXf coeffs(m+1);
	VectorXf error = _autoregression(squared_residuals, m, coeffs);


	// Compute statistic
	// Null : all coefficients = 0
	double mean = squared_residuals.mean();
	double ssr_0 = 0;
	for (int t = m; t < squared_residuals.size(); t++)
		ssr_0 += pow(squared_residuals(t) - mean,2.0);

	double ssr_1 = 0;
	for (int t = m; t < squared_residuals.size(); t++)
		ssr_1 += pow(error(t-m),2.0);

	double F = (ssr_0 - ssr_1) / m;
	F /= (ssr_1 / (squared_residuals.size() - (2*m) - 1));

	// Get p value
	Distributions pdf;
	double p_value = pdf.chi_squared(F, m);

	// Get significance
	if (p_value <= (alpha/2.0)) {
		return true;
	} else {
		return false;
	}

}



/* Pick Params */
tuple<int,int> ARFIMA::pick_parameters(const int& max_p, const int& max_q) {
	// Find the optimal p (AR coefficients) and q (MA coefficients) using AIC
	double lowest_aic = numeric_limits<double>::max();
	double test_aic;

	for (int test_p = 0; test_p <= max_p; test_p++) {
		for (int test_q = 0; test_q <= max_q; test_q++) {
			test_aic = AIC(test_p,test_q);
			if (test_aic < lowest_aic){
				lowest_aic = test_aic;
				optimal_p = test_p;
				optimal_q = test_q;
			}
		}
	}
	aic = lowest_aic;

	return make_tuple(optimal_p,optimal_q);
}


float ARFIMA::pick_d(const VectorXf& _time_series,const float& max_d) {
	// Find the lowest d (can be fractional) s.t. the time series is weakly stationary
	bool test;
	for (float _d = 0; _d <= max_d; _d+= 0.1) {
		test = unit_root_test(difference(_time_series,_d)); // dickey-fuller
		if (test) {
			optimal_d = _d;
			return _d;
		}
	}
	throw invalid_argument( "Increase max d." );
}


// Differencing and LT Memory
bool ARFIMA::unit_root_test(
	const VectorXf& _time_series,
	const double& alpha) const {
	// Unit root test to test for a random walk - Dickey Fuller
	// Returns whether not random walk is significant (phi_1 less than 1 -> not random walk)

	VectorXf test_coeffs(2);
	VectorXf test_resid = _autoregression(_time_series, 1, test_coeffs);

	double var_resid = pow(_time_series(0),2.0);
	for (int i = 1; i < test_resid.size(); i++)
		var_resid += pow(_time_series(i) - (test_coeffs(1)*_time_series(i-1)),2.0);
	var_resid /= test_resid.size();

	double bot_sum = 0;
	for (int i = 1; i < test_resid.size(); i++)
		bot_sum = pow(_time_series(i-1),2.0);

	double t_ratio = abs(test_coeffs(1) - 1) / pow(var_resid*bot_sum,0.5);

	// Calculate P values
	Distributions pdf;
	double p_value = pdf.student_t(t_ratio, _time_series.size()-1.0);

	// Get significance
	if (p_value <= (alpha/2.0)) {
		return true;
	} else {
		return false;
	}

}

VectorXf ARFIMA::difference(
	const VectorXf& _time_series,
	const float& d) {
	// Fractionally difference a time series
	// Uses a fixed-width window

	// 1. Create our weight vector w
	differencing_weights = _get_weights(_time_series.size(), d);

	// 2. transform data
	VectorXf differenced_series(_time_series.size() - k_weights);
	for (int t = 0; t < _time_series.size()-k_weights; t++)
		differenced_series(t) = differencing_weights.dot(_time_series.segment(t,k_weights));

	return differenced_series;
}

VectorXf ARFIMA::undifference_prediction(
	const VectorXf& orig_prices,
	const VectorXf& differenced_predictions,
	const float& d) {
	// Undifference time series we have already differenced
	if (k_weights > orig_prices.size())
		throw invalid_argument( "Time Series too small." );

	VectorXf weights = differencing_weights.tail(k_weights-1);
	double first_weight = differencing_weights(0);

	VectorXf undifferenced_predictions(differenced_predictions.size());
	double past_diff;
	int print_count = 0;
	for (int t = 0; t < undifferenced_predictions.size(); t++) {
		past_diff = weights.dot(orig_prices.segment(t,k_weights-1));
		undifferenced_predictions(t) = (differenced_predictions(t) - past_diff)/first_weight;

		if ((print_count > 0) && (print_count < 10)) {
			cout << differenced_predictions(t) << '\t' << orig_prices(t+k_weights) + past_diff << endl;
		}
		print_count++;
	}

	return undifferenced_predictions;

}

VectorXf ARFIMA::_get_weights(
	const int& time_series_size,
	const float& d,
	const float& weight_cutoff,
	const int& max_weights_vector_size) {
	// Get weights for fractional differencing

	int max_size = min(max_weights_vector_size, (int)(2.0 * time_series_size / 3.0));
	VectorXf w(max_size);
	w(0) = 1.0;

	k_weights = 1;
	while ((k_weights < max_size) && (abs(w(k_weights-1)) > weight_cutoff)) {
		w(k_weights) = -w(k_weights-1) * (d - k_weights + 1) / k_weights;
		k_weights++;
	}
	return w.head(k_weights);
}


double ARFIMA::rescaled_range(const int& start, const int& end) const {
	// lo modified rescaled range

	// 1. mean
	VectorXf ts_segment = time_series.segment(start, end);
	double mean = ts_segment.mean();

	// 2. mean adjusted series
	VectorXf y = ts_segment - (VectorXf::Ones(ts_segment.size())*mean);

	// 3. Cumulative deviate series
	VectorXf z(y.size());
	for (int t = 0; t < y.size(); t++) {
		z(t) = 0;
		for (int i = 0; i <= t; i++)
			z(t) += y(i);
	}

	// 4. range
	double R = z.maxCoeff() - z.minCoeff();

	// 5. adjusted standard deviation
	double var = 0;
	for (int i = 0; i < ts_segment.size(); i++)
		var += pow(ts_segment(i) - mean,2.0);
	var /= (ts_segment.size()-1);

	// adjust
	AutoCorrelation ac;
	int q = 10;
	VectorXf autocorrs(q);
	for (int i = 0; i < q; i++)
		autocorrs(i) = ac.autocorrelation(ts_segment, i+1);

	int sum = 0;
	for (int i = 0; i < q; i++)
		sum += (1 - ((i+1)/(q+1))) * autocorrs(i);

	double lo_var = var + 2 * sum;
	double lo_sd = pow(lo_var, 0.5);

	// 6. R/S
	return R / lo_sd;
}







#endif