#ifndef __HMM_GMM_H
#define __HMM_GMM_H

#include <iostream>
#include <vector>
#include <string>
#include <eigen/Eigen/Dense>
using namespace std;
using namespace Eigen;


class HMM_GMM {
	// Uses a gibbs sampler to estimate the hidden states, 
	// gaussian parameters of how the observed data is generated,
	// and the transition probabilities between 2 states
private:
	VectorXf observed;

	MatrixXf hidden_states_samples;
	MatrixXf dist_means_samples;
	MatrixXf dist_sds_samples;
	MatrixXf transition_samples;

	VectorXf optimal_hidden_states;
	VectorXf hidden_states_probabilities;
	VectorXf optimal_means;
	VectorXf optimal_sds;
	VectorXf optimal_transitions;

	/* HELPERS */
	double p_state_given_other_state(
		const int& state_on,
		const int& state_other,
		const double& transition_on) const;

	int transition_prob(
		const int& state,
		const double& t0,
		const double& t1) const;

	int sample_state(
		const int& state_index,
		const int& previous_hidden_state,
		const int& next_hidden_state,
		const double& dist_mean_s0,
		const double& dist_mean_s1,
		const double& dist_sd_s0,
		const double& dist_sd_s1,
		const double& transition_s0,
		const double& transition_s1,
		const string& first_last = "middle") const;

	double sample_transition_probability(
		const int& sample_state,
		const double& transition_s0,
		const double& transition_s1,
		const VectorXf& hidden_states,
		const double& step_size,
		const double& trans_alpha,
		const double& trans_beta,
		bool& accept) const;

	double sample_sd(
		const double& sd_orig,
		const int& sample_state,
		const double& state_mean,
		const VectorXf& hidden_states,
		const double& inv_precision,
		const double& sd_confidence,
		const double& mean_mean,
		const double& mean_confidence,
		const double& step_size,
		bool& accept) const;

	double sample_mean(
		const double& mean_orig,
		const int& sample_state,
		const double& state_sd,
		const VectorXf& hidden_states,
		const double& mean_mean,
		const double& mean_confidence,
		const double& step_size,
		bool& accept) const;

	void find_hidden_states(
		const VectorXf& transitions,
		const VectorXf& means,
		const VectorXf& sds);

	void argmax(
		const double& val1,
		const double& val2,
		double& max_val,
		int& arg) const;

public:
	/* DEFAULT */
	HMM_GMM(const VectorXf& observed_data);
	virtual ~HMM_GMM();

	/* SAMPLER */
	void gibbs_sampler(
		VectorXf& hidden_states, // must be 0s and 1s 
		double& dist_mean_s0,
		double& dist_mean_s1,
		double& dist_sd_s0,
		double& dist_sd_s1,
		double& transition_s0,
		double& transition_s1,
		const unsigned int& samples,
		const unsigned int& burn_in,
		const double& inv_precision_s0,
		const double& inv_precision_s1,
		const double& sd_confidence,
		const double& mean_mean_s0,
		const double& mean_mean_s1,
		const double& mean_confidence,
		const double& init_step_size_mean = 0.1,
		const double& init_step_size_sd = 0.1,
		const double& init_step_size_transition = 0.1,
		const double& trans_alpha = 2,
		const double& trans_beta = 5);

	/* GET PARAMS */
	void get_params();

};







#endif