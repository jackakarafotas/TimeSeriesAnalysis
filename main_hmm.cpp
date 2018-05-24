#ifndef __MAIN_HMM_CPP
#define __MAIN_HMM_CPP

#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include <string>
#include <math.h>
using namespace std;
using namespace Eigen;

#include "parser.h"
#include "stock.cpp"
#include "hmm_gmm.cpp"

string print_bool(const bool& some_bool) {
	if (some_bool)
		return "True";
	else
		return "False";
}

int main(int argc, char* argv[]) {
	string ticker = argv[1];
	int k_days = stoi(argv[2]);
	int samples = stoi(argv[3]);
	int burn_in = stoi(argv[4]);

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

	// Init vars for sampler
	// Means and SDs
	VectorXf l_returns = stock.k_period_log_returns(k_days, false);
	for (int i = 0; i < l_returns.size(); i++) {
		double ret_adj = l_returns(i) * 100.0;
		l_returns(i) = ret_adj;
	}

	double ret_mean = l_returns.mean();
	double ret_var = 0;
	for (int i = 0; i < l_returns.size(); i++) {
		ret_var += pow(l_returns(i) - ret_mean, 2.0);
	}
	ret_var /= (l_returns.size() - 1);
	double ret_sd = pow(ret_var,0.5);

	double mean_s0 = ret_mean + (0.5*ret_sd);
	double mean_s1 = ret_mean - (0.5*ret_sd);
	double sd_s0 = ret_sd - (0.2 * ret_sd);
	double sd_s1 = ret_sd + (0.2 * ret_sd);

	// Transition probabilities
	default_random_engine generator;
	uniform_real_distribution<double> sample_uniform(0.1,0.6);
	double transition_s0 = sample_uniform(generator);
	double transition_s1 = sample_uniform(generator);

	// Hidden States
	uniform_int_distribution<int> sample_discrete(0,1);
	VectorXf init_hidden_states(l_returns.size());
	for (int t = 0; t < l_returns.size(); t++)
		init_hidden_states(t) = sample_discrete(generator);


	/* SAMPLER */
	HMM_GMM hidden_markov_model(l_returns);
	hidden_markov_model.gibbs_sampler(
		init_hidden_states,
		mean_s0,
		mean_s1, 
		sd_s0, 
		sd_s1, 
		transition_s0,
		transition_s1,
		samples,
		burn_in,
		ret_sd,
		ret_sd,
		0.5,
		ret_mean,
		ret_mean,
		0.5,
		ret_sd,
		ret_sd * 0.3,
		0.1,
		2,
		5);

	hidden_markov_model.estimate_params();
	hidden_markov_model.print_params();

	return 0;
}

#endif