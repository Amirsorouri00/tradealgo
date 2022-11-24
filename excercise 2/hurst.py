# import matplotib.pyplot as plt
import numpy as np
# import pandas as pd
import yfinance as yf


def hurst(price, min_lag=2, max_lag=100):
    lags = np.arange(min_lag, max_lag + 1)
    tau = [np.std(np.subtract(price[lag:], price[:-lag]))
           for lag in lags]
    print(lags, tau)
    m = np.polyfit(np.log10(lags), np.log10(tau), 1)
    return m, lags, tau


N = 10000
rand = np.cumsum(np.random.randn(N) + 0.01)
mr = np.cumsum(np.sin(np.linspace(0, N/3*np.pi, N))/2 + 1)
tr = np.cumsum(np.arange(N)/N)

m_rand, lag_rand, rs_rand = hurst(rand)
m_mr, lag_mr, rs_mr = hurst(mr)
m_tr, lag_tr, rs_tr = hurst(tr)

print(f"Hurst(Random):\t{m_rand[0]:.3f}")
print(f"Hurst(MR):\t{m_mr[0]:.3f}")
print(f"Hurst(TR):\t{m_tr[0]:.3f}")


tickers = ['CHF=X', 'BTC-USD', 'SPY', 'GLD', 'USO']
start = '2010-01-01'
end = '2021-12-31'
colors = ['red', 'green']
yfObj = yf.Tickers(tickers)
df = yfObj.history(start=start, end=end)
df.drop(['Stock Splits', 'Dividends', 'Volume',
         'Open', 'High', 'Low'], axis=1, inplace=True)
df.columns = df.columns.swaplevel()

vals = {c[0]: hurst(df[c].dropna().values) for c in df.columns}

print(vals.keys())
# def plotHurst(m, x, y, series, name):
#     fig, ax = plt.subplots(1, 2, figsize=(15, 6))
#     ax[0].plot(np.log10(x), m[0] * np.log10(x) + m[1])
#     ax[0].scatter(np.log10(x), np.log10(y), c=colors[1])
#     ax[0].set_title(f"{name} (H = {m[0]:.3f})")
#     ax[0].set_xlabel(r"log($\tau$)")
#     ax[0].set_ylabel(r"log($\sigma_\tau$)")

#     ax[1].plot(series)
#     ax[1].set_title(f"{name}")
#     ax[1].set_ylabel("Price ($)")
#     ax[1].set_xlabel("Date")

#     return fig, ax


for k, v in vals.items():
    print(v[0][0])
# fig, ax = plotHurst(*v, df[k], k)
# plt.show()
