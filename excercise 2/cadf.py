import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat


def adf(ts, maxlag=1):
    """
    Augmented Dickey-Fuller unit root test
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)

    # Get the dimension of the array
    nobs = ts.shape[0]

    # Calculate the discrete difference
    tsdiff = np.diff(ts)

    # Create a 2d array of lags, trim invalid observations on both sides
    tsdall = lagmat(tsdiff[:, None], maxlag, trim='both', original='in')
    # Get dimension of the array
    nobs = tsdall.shape[0]

    # replace 0 xdiff with level of x
    tsdall[:, 0] = ts[-nobs - 1:-1]
    tsdshort = tsdiff[-nobs:]

    # Calculate the linear regression using an ordinary least squares model
    results = OLS(tsdshort, add_trend(tsdall[:, :maxlag + 1], 'c')).fit()
    adfstat = results.tvalues[0]

    # Get approx p-value from a precomputed table (from stattools)
    pvalue = mackinnonp(adfstat, 'c', N=1)
    return pvalue


def cadf(x, y):
    """
    Returns the result of the Cointegrated Augmented Dickey-Fuller Test
    """
    # Calculate the linear regression between the two time series
    ols_result = OLS(x, y).fit()

    # Augmented Dickey-Fuller unit root test
    return adf(ols_result.resid)
