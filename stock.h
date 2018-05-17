#ifndef __STOCK_H
#define __STOCK_H

#include <iostream>
#include <string>
#include <vector>
#include <eigen/Eigen/Dense>
using namespace std;
using namespace Eigen;

class Stock {
private:
	vector<string> dates;
	VectorXf price_open;
	VectorXf price_close;
	VectorXf price_high;
	VectorXf price_low;
	VectorXf volume;
public:
	VectorXf simple_returns;
	VectorXf log_returns;

	// Constructor
	Stock(
		const vector<string>& _dates,
		const vector<double>& _price_open,
		const vector<double>& _price_close,
		const vector<double>& _price_high,
		const vector<double>& _price_low,
		const vector<double>& _volume);

	// Virtual Destructor
	virtual ~Stock();

	// Main methods
	// returns
	VectorXf k_period_log_returns(
		const unsigned int& k,
		const bool& preserve_difference = true) const;

	VectorXf k_period_simple_returns(
		const unsigned int& k,
		const bool& preserve_difference = true) const;

	// Prices
	VectorXf k_period_log_prices(const unsigned int& k) const;

	// statistics
	double mean(
		const VectorXf& returns) const;

	double variance(
		const VectorXf& returns) const;

	double skewness(
		const VectorXf& returns) const;

	double excess_kurtosis(
		const VectorXf& returns) const;

	bool jb_test(
		const double& skewness,
		const double& excess_kurtosis,
		const double& T,
		const double& alpha = 0.05) const;

	// Getters
	vector<string> get_dates() const;
	VectorXf get_price_open() const;
	VectorXf get_price_close() const;
	VectorXf get_price_high() const;
	VectorXf get_price_low() const;
	VectorXf get_volume() const;
	VectorXf get_simple_returns() const;
	VectorXf get_log_returns() const;

	// Printer
	void print_statistics(
		const VectorXf& returns) const;
};





#endif