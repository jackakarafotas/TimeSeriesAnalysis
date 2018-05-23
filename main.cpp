#ifndef __MAIN_CPP
#define __MAIN_CPP

#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include <string>
#include <math.h>
using namespace std;
using namespace Eigen;

#include "parser.h"
#include "stock.cpp"
#include "autocorrelation.cpp"
#include "arfima.cpp"

string print_bool(const bool& some_bool) {
	if (some_bool)
		return "True";
	else
		return "False";
}

int min_int(const int& a, const int& b) {
	if (a < b)
		return a;
	else
		return b;
}

int main(int argc, char* argv[]) {
	string ticker = argv[1];
	int k_days = stoi(argv[2]);
	string folder = "stock_dataframes/";
	string ending = ".csv";

	string file_path;
	file_path.append(folder);
	file_path.append(ticker);
	file_path.append(ending);

	Parser parser;
	vector<string> dates;
	vector<double> price_open;
	vector<double> price_close;
	vector<double> price_high;
	vector<double> price_low;
	vector<double> volume;

	parser.parse_data(
		file_path,
		dates,
		price_open,
		price_close,
		price_high,
		price_low,
		volume);


	Stock stock(
		dates,
		price_open,
		price_close,
		price_high,
		price_low,
		volume);

	cout << '\n' << endl;
	stock.print_statistics(stock.log_returns);

	VectorXf l_returns = stock.k_period_log_returns(k_days, false);
	VectorXf log_close = stock.k_period_log_prices(k_days);

	// Look at autocorrelations
	AutoCorrelation autocorr;
	cout << "\nAUTOCORRELATIONS (log returns):" << endl;
	for (int i = 1; i <= 10; i++) {
		cout << "Lag: " << i 
		<< "\tSignficiant? " << autocorr.test_ac_lag(l_returns,i)
		<< "\tLjung-Box: " << autocorr.test_autocorrelation(l_returns, i, i)
		<< "\tAC: " << autocorr.autocorrelation(l_returns, i) << endl;
	}


	// ARFIMA
	cout << "\nARFIMA MODEL" << endl;
	ARFIMA* model = new ARFIMA();
	cout << "Time Series not a Random Walk? " << print_bool(model->unit_root_test(log_close)) << endl;

	// Difference
	float d = model->pick_d(log_close,2);
	VectorXf differenced_ts = model->difference(log_close, d);
	model->set_time_series(differenced_ts);

	VectorXf u_ts = model->undifference_prediction(log_close, differenced_ts, d);
	for (int i = 0; i < 20; i++) {
		cout << "ORIG: " << log_close(i)
		<< "\tTRANSFORMED: " << u_ts(i) << endl;
	}

	// Find optimal p and q
	model->train_test_split();
	tuple<int,int> p_q = model->pick_parameters(10,2);
	cout << "Optimal p: " << get<0>(p_q) 
	<< "\nOptimal q: " << get<1>(p_q)
	<< "\nOptimal d: " << d << endl;

	// TRAIN
	model->train(get<0>(p_q),get<1>(p_q));
	vector<bool> significant = model->test_coefficient_significance();

	cout << "\nARMA Coefficients: " << endl;
	for (int i = 0; i < significant.size(); i++)
		cout << model->coefficients(i) << "\tSIGNIFICANT: " << print_bool(significant[i]) << endl; 

	model->validate();
	VectorXf upper_predict(model->validation_pred.size());
	VectorXf lower_predict(model->validation_pred.size());
	model->predict_bounds(model->validation_pred, 0.95, upper_predict, lower_predict);

	int orig_validation_size = model->validation.size() + model->k_weights;
	VectorXf actual_log_prices = log_close.tail(orig_validation_size);
	VectorXf differenced_validation = model->difference(actual_log_prices, d).tail(model->validation_comp.size());

	VectorXf validation_log_close = log_close.segment(log_close.size()-orig_validation_size,orig_validation_size-1);
	VectorXf log_price_prediction = model->undifference_prediction(validation_log_close, model->validation_pred, d);
	cout << "UNDIFF COMP" << endl;
	VectorXf validation_log_prices = model->undifference_prediction(validation_log_close, model->validation_comp, d);

	cout << "\nPREDICTION:" << endl;
	cout << "Test MSE: " << model->MSE(model->validation_pred,model->validation_comp) << endl;
	for (int i = 0; i < min_int(20,model->validation_pred.size()); i++) {
		cout << "Actual: " << model->validation_comp(i)
		<< "\tPredicted: " << model->validation_pred(i)
		<< "\tDifferenced: " << differenced_validation(i)
		<< "\tLower: " << lower_predict(i)
		<< "\tUpper: " << upper_predict(i) << endl;
	}
	for (int i = 1; i <= min_int(20,model->validation_pred.size()); i++) {
		cout << "Actual price: " << actual_log_prices(actual_log_prices.size()-i)
		<< "\tUndifferenced Comp Prices: " << (validation_log_prices(validation_log_prices.size()-i))
		<< "\tPredicted price: " << (log_price_prediction(log_price_prediction.size() - i)) << endl;
	}

	cout << "\nSIGNIFICANT AUTOCORRELATION IN RESIDUALS? : " << print_bool(model->test_residual_autocorrelation(4)) << endl;
	cout << "R2: " << model->R2() << endl;
	cout << "ADJUSTED R2: " << model->adjusted_R2() << endl;
	cout << "Displayes ARCH effects? " << print_bool(model->test_arch_effect(5)) << endl;;

	return 0;
}


#endif