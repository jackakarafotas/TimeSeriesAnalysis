ARFIMA model for log stock prices: 
- Fractionally differences a stock price (uses a fixed window)
- Fits an ARMA model on it (using the Hannan–Rissanen algorithm)
- Tests the residual autocorrelation, whether there are ARCH effects, and coefficient significance 

Hidden Markov Model (observed data distributed according to Gaussian) - fit with Gibbs Sampler
- Fits a hidden markov model (2 hidden states, observed data is normally distributed) to time series data using a gibbs sampler
- Within the gibbs sampler used Metropolis-Hastings jumps
- Finds the optimal parameters (besides the hidden states) by finding the mean of the samples
- Finds the optimal hidden states using the Viterbi Algoirthm (implemented so that probabilities are also returned)

Beta:
- find a relationship coefficient between different time series (takes into account the need to difference the time series if the residuals are not stationary)

Implements:
- Dickey-Fuller test (unit-root test / stationarity test)
- Ljung-Box test (significant autocorrelation test)
- ARCH test (conditionally heteroskedasticity test)
- Lo-modified Rescaled Range (long term memory test)
- Multi-Regression coefficient p-test
- autocorrelation calculation
- multiple distributions