#ifndef __STOCK_CPP
#define __STOCK_CPP

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <cmath>
#include <eigen/Eigen/Dense>
#include "stock.h"
#include "distributions.cpp"
using namespace std;
using namespace Eigen;

// Helper
VectorXf std2eigen(const vector<double>& vec) {
	// Copies data into a VectorXf
	VectorXf e_vec(vec.size());

	for (int i = 0; i < vec.size(); i++)
		e_vec(i) = vec[i];

	return e_vec;
}

// INIT
Stock::Stock(
	const vector<string>& _dates,
	const vector<double>& _price_open,
	const vector<double>& _price_close,
	const vector<double>& _price_high,
	const vector<double>& _price_low,
	const vector<double>& _volume) {
	// Assign private data
	dates = _dates;
	price_open = std2eigen(_price_open);
	price_close = std2eigen(_price_close);
	price_high = std2eigen(_price_high);
	price_low = std2eigen(_price_low);
	volume = std2eigen(_volume);

	// Calculate returns
	simple_returns.resize(price_close.size());
	log_returns.resize(price_close.size());
	simple_returns(0) = 0;
	log_returns(0) = 0;

	for (int i = 1; i < price_close.size(); i++) {
		simple_returns(i) = (price_close(i)-price_close(i-1))/price_close(i-1);
	}

	for (int i = 1; i < price_close.size(); i++) {
		log_returns(i) = log(price_close(i)) - log(price_close(i-1));
	}
}

Stock::~Stock() {}



// Returns
VectorXf Stock::k_period_log_returns(
	const unsigned int& k,
	const bool& preserve_difference) const {
	/*
	- Returns log returns for a period of size k
	- if preserve difference == true (default) it preserves 
	  close to the same amount of time steps in the original
	  time series
	  --> else: only keep every k'th log return 
	*/
	VectorXf k_log_returns(log_returns.size());

	for (int i = k; i < log_returns.size(); i++)
		k_log_returns(i) = log(price_close(i)) - log(price_close(i-k));

	if (preserve_difference) {
		return k_log_returns;
	} else {
		int new_size = floor(k_log_returns.size()/k);
		VectorXf k_return(new_size);
		for (int i = 0; i < new_size; i++)
			k_return(i) = k_log_returns(i*k);
		return k_return;
	}
}

VectorXf Stock::k_period_simple_returns(
	const unsigned int& k,
	const bool& preserve_difference) const {
	/*
	- Returns simple returns for a period of size k
	- if preserve difference == true (default) it preserves 
	  close to the same amount of time steps in the original
	  time series
	  --> else: only keep every k'th simple return 
	*/
	VectorXf k_simple_returns;

	for (int i = k; i < log_returns.size(); i++)
		k_simple_returns(i) = (price_close(i)-price_close(i-k))/price_close(i-k);

	if (preserve_difference) {
		return k_simple_returns;
	} else {
		int new_size = floor(k_simple_returns.size()/k);
		VectorXf k_return(new_size);
		for (int i = 0; i < new_size; i++)
			k_return(i) = k_simple_returns(i*k);
		return k_return;
	}
}

// Prices
VectorXf Stock::k_period_log_prices(const unsigned int& k) const {
	/*
	- Returns log closing prices for a period of size k
	*/
	int size = floor(price_close.size()/k);
	VectorXf k_prices(size);

	for (int i = 0; i < size; i++)
		k_prices(i) = log(price_close(i*k));
	return k_prices;
}


// Statistics
double Stock::mean(
	const VectorXf& returns) const {
	// Mean of daily log returns
	return returns.mean();
}

double Stock::variance(
	const VectorXf& returns) const {
	// Variance of daily log returns
	double var = 0;
	double m = returns.mean();
	for (int i = 0; i < returns.size(); i++)
		var += pow((returns(i) - m),2.0);

	var /= ((double) returns.size() - 1.0);
	return var;
}

double Stock::skewness(
	const VectorXf& returns) const {
	// Skewness of daily log returns
	double sum = 0;
	double m = returns.mean();
	double sd_3 = pow(variance(returns),1.5);

	for (int i = 0; i < returns.size(); i++)
		sum += pow((returns(i) - m),3.0);

	return sum / (((double) returns.size() - 1.0) * sd_3);

}

double Stock::excess_kurtosis(
	const VectorXf& returns) const {
	// Excess Kurtosis of daily log returns
	double sum = 0;
	double m = returns.mean();
	double var = variance(returns);

	for (int i = 0; i < returns.size(); i++)
		sum += pow((returns(i) - m),4.0);

	return (sum / (((double) returns.size() - 1.0) * pow(var,2.0))) - 3.0;

}

bool Stock::jb_test(
	const double& skewness,
	const double& excess_kurtosis,
	const double& T,
	const double& alpha) const {
	// Tests normality of rt (null = log returns are normally distributed)

	double jb_statistic = pow(skewness,2.0) / (6.0 / (double)T);
	jb_statistic += pow(excess_kurtosis,2.0) / (24.0 / T);

	Distributions pdf;
	double p_value = pdf.chi_squared(jb_statistic, 2.0);

	if (p_value <= alpha) {
		return true; // Null Hypothesis is rejected
	} else {
		return false; // Null Hypothesis is accepted
	}

}


void Stock::print_statistics(
	const VectorXf& returns) const {
	double avg = mean(returns);
	double var = variance(returns);
	double skew = skewness(returns);
	double e_kurt = excess_kurtosis(returns);

	cout << "MEAN: " << avg << endl;
	cout << "VARIANCE: " << var << endl;
	cout << "STANDARD DEVIATION: " << pow(var,0.5) << endl;
	cout << "SKEWNESS: " << skew << endl;
	cout << "EXCESS KURTOSIS: " << e_kurt << endl;

	bool normality_test = jb_test(
		skew,
		e_kurt,
		returns.size());

	if (normality_test == 0) {
		cout << "RETURNS ARE NORMAL" << endl;
	} else if (normality_test == 1) {
		cout << "RETURNS ARE NOT NORMAL" << endl;
	}
}


// Getters
vector<string> Stock::get_dates() const { return dates;}
VectorXf Stock::get_price_open() const { return price_open;}
VectorXf Stock::get_price_close() const { return price_close;}
VectorXf Stock::get_price_high() const { return price_high;}
VectorXf Stock::get_price_low() const { return price_low;}
VectorXf Stock::get_volume() const { return volume;}
VectorXf Stock::get_simple_returns() const { return simple_returns;}
VectorXf Stock::get_log_returns() const { return log_returns;}




#endif