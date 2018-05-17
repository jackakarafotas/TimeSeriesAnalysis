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


	// Train ARFIMA model
	cout << "\nARFIMA MODEL" << endl;
	unsigned int cutoff = floor(3*log_close.size()/4);
	VectorXf data_train = log_close.head(cutoff);
	VectorXf data_test = log_close.tail(log_close.size()-cutoff);

	ARFIMA* model = new ARFIMA(data_train);
	cout << "Time Series not a Random Walk? " << print_bool(model->unit_root_test(data_train)) << endl;

	// Pick params
	float d = model->pick_d(5);
	tuple<int,int> p_q = model->pick_parameters(10,2);
	cout << "Optimal p: " << get<0>(p_q) 
	<< "\nOptimal q: " << get<1>(p_q)
	<< "\nOptimal d: " << d << endl;
	

	model->train(get<0>(p_q),d,get<1>(p_q));
	vector<bool> significant = model->test_coefficient_significance();

	cout << "\nARMA Coefficients: " << endl;
	for (int i = 0; i < significant.size(); i++)
		cout << model->coefficients(i) << "\tSIGNIFICANT: " << print_bool(significant[i]) << endl; 

	VectorXf prediction = model->predict(data_test);
	VectorXf upper_predict(prediction.size());
	VectorXf lower_predict(prediction.size());
	VectorXf diff_actual = model->difference(data_test,d);
	VectorXf actual = diff_actual.tail(diff_actual.size()-model->adj_size);
	model->predict_bounds(prediction, 0.95, upper_predict, lower_predict);

	cout << "\nPREDICTION:" << endl;
	cout << "Test MSE: " << model->MSE(prediction,actual) << endl;
	for (int i = 0; i < min_int(20,prediction.size()); i++) {
		cout << "Actual: " << actual(i)
		<< "\tPredicted: " << prediction(i) 
		<< "\tLower: " << lower_predict(i)
		<< "\tUpper: " << upper_predict(i) << endl;
	}

	cout << "\nSIGNIFICANT AUTOCORRELATION IN RESIDUALS? : " << print_bool(model->test_residual_autocorrelation(4)) << endl;
	cout << "R2: " << model->R2() << endl;
	cout << "ADJUSTED R2: " << model->adjusted_r2() << endl;
	cout << "Displayes ARCH effects? " << print_bool(model->test_arch_effect(5)) << endl;;

	return 0;
}


#endif