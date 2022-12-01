import datetime
import math
from math import ceil, log

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat

SUB_SP_INDICES = ['IR', 'INTC', 'ICE', 'IP', 'IPG',
                  'IFF', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV']

SP_INDICES = ['MMM', 'AOS', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADM', 'ADBE', 'ADP', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AMD', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'AZO', 'AVB', 'AVY', 'BKR', 'BALL', 'BAC', 'BBWI', 'BAX', 'BDX', 'WRB', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISH', 'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FAST', 'FRT', 'FDX', 'FITB', 'FRC', 'FE', 'FIS', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GEN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY',
              'IR', 'INTC', 'ICE', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OGN', 'OTIS', 'PCAR', 'PKG', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PKI', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SBNY', 'SPG', 'SWKS', 'SJM', 'SNA', 'SEDG', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'VICI', 'V', 'VNO', 'VMC', 'WAB', 'WBA', 'WMT', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WRK', 'WY', 'WHR', 'WMB', 'WTW', 'GWW', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

RESOURCE = SP_INDICES

MEASURE = 'Close'


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


def merge(ts, maxlag=1):
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
    return tsdshort, tsdall


def adf(ts, maxlag=1):
    """
    Augmented Dickey-Fuller unit root test
    """
    tsdshort, tsdall = merge(ts, maxlag)

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


start = '2020-11-01'
end = '2021-11-01'
full_stock_data = yf.download(RESOURCE, start, end)

stationary_series = []
cnt = 0
for i in RESOURCE:
    if cnt == (len(RESOURCE)/2)+1:
        break
    cnt += 1
    for j in RESOURCE:
        if i == j:
            continue
        try:
            if cadf(full_stock_data[MEASURE][i], full_stock_data[MEASURE][j]) <= 0.05:
                stationary_series.append([i, j])
        except:
            continue

print(stationary_series)


def hurst(price, min_lag=2, max_lag=100):
    lags = np.arange(min_lag, max_lag + 1)
    tau = [np.std(np.subtract(price[lag:], price[:-lag]))
           for lag in lags]
    m = np.polyfit(np.log10(lags), np.log10(tau), 1)
    return m, lags, tau


def half_life(z_array):
    z_lag = np.roll(z_array, 1)
    z_lag[0] = 0
    z_ret = z_array - z_lag
    z_ret[0] = 0

    # adds intercept terms to X variable for regression
    z_lag2 = add_constant(z_lag)

    model = OLS(z_ret, z_lag2)
    res = model.fit()

    halflife = -log(2) / res.params[1]
    return halflife


hurst_dict = {}
for st in stationary_series:
    if len(hurst_dict) == 100:
        break
    df1 = full_stock_data[MEASURE][st[0]]
    df2 = full_stock_data[MEASURE][st[1]]
    ols_result = OLS(df1, df2).fit()
    df3, _ = merge(ols_result.resid)

    m_mr, lag_mr, rs_mr = hurst(df3)
    hl = half_life(df3)
    print(st, m_mr[0], hl)
    if (m_mr[0] <= 0.4 and m_mr[0] >= 0 and hl < 50):
        hurst_dict[st[0]+':'+st[1]] = [m_mr[0], hl]


def Z_Score(values, n, lookback):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    series = pd.Series(values)
    return (series - series.rolling(n).mean())/series.rolling(lookback).std()


lookback = 30


class Z_Score_Naive(Strategy):
    threshold = 2
    stoploss = 0.001

    def init(self):
        self.ZScore = self.I(Z_Score, self.data.Close,
                             self.threshold, lookback)

    def next(self):
        if (self.position.is_long) & (self.ZScore > 0):
            self.position.close()

        if (self.position.is_short) & (self.ZScore < 0):
            self.position.close()

        if self.position.pl_pct < -self.stoploss:
            self.position.close()

        if (self.ZScore < -self.threshold) & (~self.position.is_long):
            self.position.close()
            self.buy()

        if (self.ZScore > self.threshold) & (~self.position.is_short):
            self.position.close()
            self.sell()


def mdd(prices: list):
    maxDif = 0
    start = prices[0]
    for i in range(len(prices)):
        maxDif = min(maxDif, prices[i]-start)
        start = max(prices[i], start)
    return abs(maxDif)


for key, value in hurst_dict.items():
    tickers = key.split(':')
    # df1 = full_stock_data[MEASURE][tickers[0]]
    # df2 = full_stock_data[MEASURE][tickers[1]]
    asdf = yf.download(tickers[0], start, end)
    asdf2 = yf.download(tickers[1], start, end)

    ols_result = OLS(asdf, asdf2).fit()
    s = Backtest(pd.DataFrame(ols_result.resid), Z_Score_Naive)
    stats = s.run()
    print(tickers, stats)
