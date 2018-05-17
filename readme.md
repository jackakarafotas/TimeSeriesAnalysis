ARIMA model for log stock prices: 
- Fractionally differences a stock price (uses a fixed window)
- Fits an ARMA model on it (using the Hannan–Rissanen algorithm)
- Tests the residual autocorrelation, whether there are ARCH effects, and coefficient significance 

Beta:
- find a relationship coefficient between different time series (takes into account the need to difference the time series if the residuals are not stationary)

Implements:
- Dickey-Fuller test (unit-root test / stationarity test)
- Ljung-Box test (significant autocorrelation test)
- ARCH test (conditionally heteroskedasticity test)
- Lo-modified Rescaled Range (long term memory test)
- Multi-Regression coefficient p-test