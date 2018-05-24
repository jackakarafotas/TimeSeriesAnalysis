#ifndef __HMM_GMM_CPP
#define __HMM_GMM_CPP

#include "hmm_gmm.h"
#include "distributions.cpp"
#include <random>
#include <algorithm>

/* DEFAULT */
HMM_GMM::HMM_GMM(const VectorXf& observed_data) : observed(observed_data) {}
HMM_GMM::~HMM_GMM() {}

/* SAMPLER */
void HMM_GMM::gibbs_sampler(
	VectorXf& hidden_states,
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
	const double& init_step_size_mean,
	const double& init_step_size_sd,
	const double& init_step_size_transition,
	const double& trans_alpha,
	const double& trans_beta) {	
	/*
	Samples hidden states, gaussian parameters and transition probabilities 
	Assumes:
	- transition probability is distributed according to a beta distribution
	- variance is distributed according to an inverse-gamma distribution
	- mean given variance is distributed according to a normal distribution
	- the observations given the states and params are normally distributed
	*/
	// Error Catches
	if (hidden_states.size() != observed.size())
		throw invalid_argument( "Hidden States and Observed Values must be of the same size." );

	// Init
	int T = hidden_states.size();
	hidden_states_samples.resize(samples,T);
	dist_means_samples.resize(samples,2);
	dist_sds_samples.resize(samples,2);
	transition_samples.resize(samples,2);

	double step_size_transition = init_step_size_transition;
	double step_size_mean = init_step_size_mean;
	double step_size_sd = init_step_size_sd;
	double number_t_step_accept = 0;
	double number_sd_step_accept = 0;
	double number_mean_step_accept = 0;

	// Burn in
	for (int i = 0; i < burn_in; i++) {

		/* SAMPLE HIDDEN STATES */
		// S{0} - first one
		hidden_states(0) = sample_state(
			0,
			-1,
			hidden_states(1),
			dist_mean_s0,
			dist_mean_s1,
			dist_sd_s0,
			dist_sd_s1,
			transition_s0,
			transition_s1,
			"first");

		// S{2} - S{T-2}
		for (int t = 1; t < T-1; t++) {
			hidden_states(t) = sample_state(
				t,
				hidden_states(t-1),
				hidden_states(t+1),
				dist_mean_s0,
				dist_mean_s1,
				dist_sd_s0,
				dist_sd_s1,
				transition_s0,
				transition_s1,
				"middle");
		}

		// S{T-1} - last one
		hidden_states(T-1) = sample_state(
			T-1,
			hidden_states(T-2),
			-1,
			dist_mean_s0,
			dist_mean_s1,
			dist_sd_s0,
			dist_sd_s1,
			transition_s0,
			transition_s1,
			"last");


		/* SAMPLE TRANSITION PROBABILITIES*/
		bool accept_t0;
		double t0 = sample_transition_probability(
			0,
			transition_s0,
			transition_s1,
			hidden_states,
			step_size_transition,
			trans_alpha,
			trans_beta,
			accept_t0);
		transition_s0 = t0;
		if (accept_t0)
			number_t_step_accept += 0.5;

		bool accept_t1;
		double t1 = sample_transition_probability(
			1,
			transition_s0,
			transition_s1,
			hidden_states,
			step_size_transition,
			trans_alpha,
			trans_beta,
			accept_t1);
		transition_s1 = t1;
		if (accept_t1)
			number_t_step_accept += 0.5;


		/* SAMPLE SDs */ 
		bool accept_sd0;
		double sd0 = sample_sd(
			dist_sd_s0,
			0,
			dist_mean_s0,
			hidden_states,
			inv_precision_s0,
			sd_confidence,
			mean_mean_s0,
			mean_confidence,
			step_size_sd,
			accept_sd0);
		dist_sd_s0 = sd0;
		if (accept_sd0)
			number_sd_step_accept += 0.5;

		bool accept_sd1;
		double sd1 = sample_sd(
			dist_sd_s1,
			1,
			dist_mean_s1,
			hidden_states,
			inv_precision_s1,
			sd_confidence,
			mean_mean_s1,
			mean_confidence,
			step_size_sd,
			accept_sd1);
		dist_sd_s1 = sd1;
		if (accept_sd1)
			number_sd_step_accept += 0.5;

		/* SAMPLE MEANS */
		bool accept_mu0;
		double mu0 = sample_mean(
			dist_mean_s0,
			0,
			dist_sd_s0,
			hidden_states,
			mean_mean_s0,
			mean_confidence,
			step_size_mean,
			accept_mu0);
		dist_mean_s0 = mu0;
		if (accept_mu0)
			number_mean_step_accept += 0.5;

		bool accept_mu1;
		double mu1 = sample_mean(
			dist_mean_s1,
			1,
			dist_sd_s1,
			hidden_states,
			mean_mean_s1,
			mean_confidence,
			step_size_mean,
			accept_mu1);
		dist_mean_s1 = mu1;
		if (accept_mu1)
			number_mean_step_accept += 0.5;

		// Adjust step sizes based on the acceptance percentages (aim = 0.5)
		if ((i % 10) == 0) {
			double perc_t_accept = number_t_step_accept / 10.0;
			number_t_step_accept = 0.0;
			if (perc_t_accept < 0.2)
				step_size_transition *= 0.75;
			else if (perc_t_accept < 0.4)
				step_size_transition *= 0.9;
			else if (perc_t_accept > 0.8)
				step_size_transition *= 1.3;
			else if (perc_t_accept > 0.6)
				step_size_transition *= 1.1;

			double perc_sd_accept = number_sd_step_accept / 10.0;
			number_sd_step_accept = 0.0;
			if (perc_sd_accept < 0.2)
				step_size_sd *= 0.75;
			else if (perc_sd_accept < 0.4)
				step_size_sd *= 0.9;
			else if (perc_sd_accept > 0.8)
				step_size_sd *= 1.3;
			else if (perc_sd_accept > 0.6)
				step_size_sd *= 1.1;

			double perc_mean_accept = number_mean_step_accept / 10.0;
			number_mean_step_accept = 0.0;
			if (perc_mean_accept < 0.2)
				step_size_mean *= 0.75;
			else if (perc_mean_accept < 0.4)
				step_size_mean *= 0.9;
			else if (perc_mean_accept > 0.8)
				step_size_mean *= 1.3;
			else if (perc_mean_accept > 0.6)
				step_size_mean *= 1.1;
		}

	}

	// Sample
	bool accept;
	for (int i = 0; i < samples; i++) {

		/* SAMPLE HIDDEN STATES */
		// S{0} - first one
		hidden_states(0) = sample_state(
			0,
			-1,
			hidden_states(1),
			dist_mean_s0,
			dist_mean_s1,
			dist_sd_s0,
			dist_sd_s1,
			transition_s0,
			transition_s1,
			"first");
		hidden_states_samples(i,0) = hidden_states(0);

		// S{2} - S{T-2}
		for (int t = 1; t < T-1; t++) {
			hidden_states(t) = sample_state(
				t,
				hidden_states(t-1),
				hidden_states(t+1),
				dist_mean_s0,
				dist_mean_s1,
				dist_sd_s0,
				dist_sd_s1,
				transition_s0,
				transition_s1,
				"middle");
			hidden_states_samples(i,t) = hidden_states(t);
		}

		// S{T-1} - last one
		hidden_states(T-1) = sample_state(
			T-1,
			hidden_states(T-2),
			-1,
			dist_mean_s0,
			dist_mean_s1,
			dist_sd_s0,
			dist_sd_s1,
			transition_s0,
			transition_s1,
			"last");
		hidden_states_samples(i,T-1) = hidden_states(T-1);


		/* SAMPLE TRANSITION PROBABILITIES*/
		double t0 = sample_transition_probability(
			0,
			transition_s0,
			transition_s1,
			hidden_states,
			step_size_transition,
			trans_alpha,
			trans_beta,
			accept);
		transition_s0 = t0;
		transition_samples(i,0) = transition_s0;

		double t1 = sample_transition_probability(
			1,
			transition_s0,
			transition_s1,
			hidden_states,
			step_size_transition,
			trans_alpha,
			trans_beta,
			accept);
		transition_s1 = t1;
		transition_samples(i,1) = transition_s1;


		/* SAMPLE SDs */ 
		double sd0 = sample_sd(
			dist_sd_s0,
			0,
			dist_mean_s0,
			hidden_states,
			inv_precision_s0,
			sd_confidence,
			mean_mean_s0,
			mean_confidence,
			step_size_sd,
			accept);
		dist_sd_s0 = sd0;
		dist_sds_samples(i,0) = dist_sd_s0;

		double sd1 = sample_sd(
			dist_sd_s1,
			1,
			dist_mean_s1,
			hidden_states,
			inv_precision_s1,
			sd_confidence,
			mean_mean_s1,
			mean_confidence,
			step_size_sd,
			accept);
		dist_sd_s1 = sd1;
		dist_sds_samples(i,1) = dist_sd_s1;

		/* SAMPLE MEANS */
		double mu0 = sample_mean(
			dist_mean_s0,
			0,
			dist_sd_s0,
			hidden_states,
			mean_mean_s0,
			mean_confidence,
			step_size_mean,
			accept);
		dist_mean_s0 = mu0;
		dist_means_samples(i,0) = dist_mean_s0;

		double mu1 = sample_mean(
			dist_mean_s1,
			1,
			dist_sd_s1,
			hidden_states,
			mean_mean_s1,
			mean_confidence,
			step_size_mean,
			accept);
		dist_mean_s1 = mu1;
		dist_means_samples(i,1) = dist_mean_s1;

	}

}


void HMM_GMM::get_params() {
	/* Finds the optimal parameter:
	-> Finds the mean for all params besides the states
	-> then uses these params and the viterbi algorithm
		to find the optimal hidden states */
	// Transitions
	optimal_transitions(0) = transition_samples.col(0).mean();
	optimal_transitions(1) = transition_samples.col(1).mean();

	// Means and SDs
	optimal_means(0) = dist_means_samples.col(0).mean();
	optimal_means(1) = dist_means_samples.col(1).mean();
	optimal_sds(0) = dist_sds_samples.col(0).mean();
	optimal_sds(1) = dist_sds_samples.col(1).mean();

	// Use these to find the optimal hidden states
	find_hidden_states(optimal_transitions, optimal_means, optimal_sds);

}


/* HELPERS */
// VITERBI
void HMM_GMM::find_hidden_states(
	const VectorXf& transitions,
	const VectorXf& means,
	const VectorXf& sds) {
	/* Uses the viterbi algorithm to find the optimal
	hidden states and their probability */

	// INIT
	Distributions dist;
	int T = observed.size();
	MatrixXf T_1(2,T);
	MatrixXf T_2(2,T);

	/* FORWARD */
	// Initial state
	double p_1 = hidden_states_samples.col(0).sum() / hidden_states_samples.rows(); // 1s will be summed
	T_1(0,0) = (1.0 - p_1) * dist.normal(observed(0),means(0),sds(0));
	T_2(0,0) = 0;
	T_1(1,0) = p_1 * dist.normal(observed(1),means(1),sds(1));
	T_2(1,0) = 0;

	// Loop states
	for (int t = 1; t < T; t++) {
		for (int i = 0; i < 2; i++) {
			double val1 = T_1(0,t-1) * p_state_given_other_state(0,i,transitions(0)) * dist.normal(observed(t),means(i),sds(i));
			double val2 = T_1(1,t-1) * p_state_given_other_state(1,i,transitions(1)) * dist.normal(observed(t),means(i),sds(i));
			double max_val;
			int arg;
			argmax(val1,val2,max_val,arg);

			T_1(i,t) = max_val;
			T_2(i,t) = arg;
		}
	}

	/* BACKWARD */
	// last state
	double max_val;
	int arg;
	argmax(T_1(0,T-1),T_1(1,T-1),max_val,arg);
	optimal_hidden_states(T-1) = arg;
	hidden_states_probabilities(T-1) = max_val;

	// loop states
	for (int t = T-1; t > 0; t--) {
		optimal_hidden_states(t-1) = T_2(optimal_hidden_states(t),t);
		hidden_states_probabilities(t-1) = T_1(optimal_hidden_states(t),t);
	}


}

// SAMPLERS
int HMM_GMM::sample_state(
		const int& state_index,
		const int& previous_hidden_state,
		const int& next_hidden_state,
		const double& dist_mean_s0,
		const double& dist_mean_s1,
		const double& dist_sd_s0,
		const double& dist_sd_s1,
		const double& transition_s0,
		const double& transition_s1,
		const string& first_last) const {
	/* Samples a state (0 or 1) for particular state-variable */
	// INIT
	Distributions dist;
	default_random_engine generator;
	uniform_real_distribution<double> sample_uniform(0.0,1.0);

	// Get probs
	double p0_unnorm = dist.normal(observed(state_index), dist_mean_s0, dist_sd_s0);
	double p1_unnorm = dist.normal(observed(state_index), dist_mean_s1, dist_sd_s1);

	if (first_last == "first") {
		p0_unnorm *= p_state_given_other_state(0,next_hidden_state,transition_s0);
		p1_unnorm *= p_state_given_other_state(1,next_hidden_state,transition_s1);
	} else if (first_last == "last") {
		p0_unnorm *= p_state_given_other_state(
			previous_hidden_state,
			0,
			transition_prob(previous_hidden_state,transition_s0,transition_s1));
		p1_unnorm *= p_state_given_other_state(
			previous_hidden_state,
			1,
			transition_prob(previous_hidden_state,transition_s0,transition_s1));
	} else if (first_last == "middle") {
		p0_unnorm *= p_state_given_other_state(0,next_hidden_state,transition_s0);
		p0_unnorm *= p_state_given_other_state(
			previous_hidden_state,
			0,
			transition_prob(previous_hidden_state,transition_s0,transition_s1));

		p1_unnorm *= p_state_given_other_state(1,next_hidden_state,transition_s1);
		p1_unnorm *= p_state_given_other_state(
			previous_hidden_state,
			1,
			transition_prob(previous_hidden_state,transition_s0,transition_s1));
	} else {
		throw invalid_argument( "Please enter first, last, or middle into first_last." );
	}

	// sample:
	double p_comb = p0_unnorm + p1_unnorm;
	double p0 = p0_unnorm / p_comb;
	double p1 = p1_unnorm / p_comb;
	double u = sample_uniform(generator);
	if (u <= p0)
		return 0;
	else
		return 1;
}

double HMM_GMM::sample_transition_probability(
	const int& sample_state,
	const double& transition_s0,
	const double& transition_s1,
	const VectorXf& hidden_states,
	const double& step_size,
	const double& trans_alpha,
	const double& trans_beta,
	bool& accept) const {
	/* Samples a transition probability for a particular state 
	using a Metropolis-Hastings step.
	Assumes the transitions are distributed according to beta*/
	double t_orig;
	if (sample_state == 0) {
		t_orig = transition_s0;
	} else if (sample_state == 1) {
		t_orig = transition_s1;
	} else {
		throw invalid_argument( "Please enter 0 or 1 for the main transition." );
	}

	// INIT
	Distributions dist;
	default_random_engine generator;
	uniform_real_distribution<double> sample_uniform(0.0,1.0);
	normal_distribution<double> sample_standard_norm(0.0,1.0);

	// 1. proposal
	double t_proposal = t_orig + (step_size*sample_standard_norm(generator));
	// 2. calculate acceptance probability
	double posterior_division = dist.beta_dist(t_proposal, trans_alpha, trans_beta);
	posterior_division /= dist.beta_dist(t_orig, trans_alpha, trans_beta);
	for (int t = 0; t < hidden_states.size()-1; t++) {
		// Multiply by the probability produced by the proposal
		// and divide by the probability produced by the original
		// --> attempts to make sure to avoid floating point errors
		if (sample_state == 0) {
			posterior_division *= p_state_given_other_state(
				hidden_states(t),
				hidden_states(t+1),
				transition_prob(hidden_states(t), t_proposal, transition_s1));
			posterior_division /= p_state_given_other_state(
				hidden_states(t),
				hidden_states(t+1),
				transition_prob(hidden_states(t), t_orig, transition_s1));
		} else if (sample_state == 1) {
			posterior_division *= p_state_given_other_state(
				hidden_states(t),
				hidden_states(t+1),
				transition_prob(hidden_states(t), transition_s0, t_proposal));
			posterior_division /= p_state_given_other_state(
				hidden_states(t),
				hidden_states(t+1),
				transition_prob(hidden_states(t), transition_s0, t_orig));
		}
	}
	double p_accept = min(1.0, posterior_division);
	// 3. sample
	double u = sample_uniform(generator);
	if (u <= p_accept) {
		accept = true;
		return t_proposal;
	}
	else {
		accept = false;
		return t_orig;
	}
}

double HMM_GMM::sample_sd(
	const double& sd_orig,
	const int& sample_state,
	const double& state_mean,
	const VectorXf& hidden_states,
	const double& inv_precision,
	const double& sd_confidence,
	const double& mean_mean,
	const double& mean_confidence,
	const double& step_size,
	bool& accept) const {
	/* Samples standard deviation using a metropolis-hastings step.
	Assumes that standard deviation is distributed according to inverse-gamma,
	and that the mean given standard deviation is normal distributed */
	if ((sample_state != 0) || (sample_state != 1))
		throw invalid_argument( "Please enter a state of 0 or 1." );

	// INIT
	Distributions dist;
	default_random_engine generator;
	uniform_real_distribution<double> sample_uniform(0.0,1.0);
	normal_distribution<double> sample_standard_norm(0.0,1.0);

	// 1. proposal
	double sd_proposal = sd_orig + (step_size*sample_standard_norm(generator));
	// 2. calculate acceptance probability
	double posterior_division = dist.normal(state_mean, mean_mean, sd_proposal / pow(mean_confidence,0.5));
	posterior_division /= dist.normal(state_mean, mean_mean, sd_orig / pow(mean_confidence,0.5));
	posterior_division *= dist.inverse_gamma(sd_proposal, sd_confidence, inv_precision);
	posterior_division /= dist.inverse_gamma(sd_orig, sd_confidence, inv_precision);

	for (int t = 0; t < hidden_states.size(); t++) {
		if (hidden_states(t) == sample_state) {
			posterior_division *= dist.normal(observed(t),state_mean,sd_proposal);
			posterior_division /= dist.normal(observed(t),state_mean,sd_orig);
		}
	}
	double p_accept = min(1.0, posterior_division);
	// 3. sample
	double u = sample_uniform(generator);
	if (u <= p_accept) {
		accept = true;
		return sd_proposal;
	}
	else {
		accept = false;
		return sd_orig;
	}
}

double HMM_GMM::sample_mean(
	const double& mean_orig,
	const int& sample_state,
	const double& state_sd,
	const VectorXf& hidden_states,
	const double& mean_mean,
	const double& mean_confidence,
	const double& step_size,
	bool& accept) const {
	/* Samples mean using a metropolis-hastings step.
	Assumes that the mean given standard deviation is normal distributed */
	if ((sample_state != 0) || (sample_state != 1))
		throw invalid_argument( "Please enter a state of 0 or 1." );

	// INIT
	Distributions dist;
	default_random_engine generator;
	uniform_real_distribution<double> sample_uniform(0.0,1.0);
	normal_distribution<double> sample_standard_norm(0.0,1.0);

	// 1. proposal
	double mean_proposal = mean_orig + (step_size*sample_standard_norm(generator));
	// 2. calculate acceptance probability
	double posterior_division = dist.normal(mean_proposal, mean_mean, state_sd / pow(mean_confidence,0.5));
	posterior_division /= dist.normal(mean_orig, mean_mean, state_sd / pow(mean_confidence,0.5));

	for (int t = 0; t < hidden_states.size(); t++) {
		if (hidden_states(t) == sample_state) {
			posterior_division *= dist.normal(observed(t),mean_proposal,state_sd);
			posterior_division /= dist.normal(observed(t),mean_orig,state_sd);
		}
	}
	double p_accept = min(1.0, posterior_division);
	// 3. sample
	double u = sample_uniform(generator);
	if (u <= p_accept) {
		accept = true;
		return mean_proposal;
	}
	else {
		accept = false;
		return mean_orig;
	}
}

double HMM_GMM::p_state_given_other_state(
	const int& state_on,
	const int& state_other,
	const double& transition_on) const {
	if ((state_other != 0) || state_other != 1)
		throw invalid_argument( "please enter proper state other (0 or 1)." );

	if (state_on == state_other)
		return 1 - transition_on;
	else if (state_on != state_other)
		return transition_on;
}

int HMM_GMM::transition_prob(
	const int& state,
	const double& t0,
	const double& t1) const {
	if (state == 0)
		return t0;
	else if (state == 1)
		return t1;
	else
		throw invalid_argument( "State needs to be 0 or 1." );
}

void HMM_GMM::argmax(
	const double& val1,
	const double& val2,
	double& max_val,
	int& arg) const {
	if (val1 < val2) {
		max_val = val2;
		arg = 1;
	} else {
		max_val = val1;
		arg = 0;
	}
}





#endif