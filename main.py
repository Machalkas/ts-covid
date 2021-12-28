print("init...")
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
print("init done.")

# Register converters to avoid warnings
pd.plotting.register_matplotlib_converters()
plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)



data=pd.read_csv("data/full_grouped.csv")
data=data.loc[data['Country/Region'] == "Russia"]
# data.drop("Country/Region",1, inplace=True)
# data.drop("WHO Region",1, inplace=True)
y=data.loc[:,["Date","Deaths"]]
y=y.set_index(["Date"])

print(type(y))
print(y)
# y.plot(figsize=(19,4))
# plt.show()
# input()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# y=y["Deaths"].values
# y=y.to_numpy()
y2=y.values
print("Y val:",y2)
# raise

# arima=""
# min_aic=9**10
# min_arima=""
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
#             results = mod.fit()
#             arima+='ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic)+"\n"
#             if results.aic<min_aic:
#                 min_aic=results.aic
#                 min_arima='ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic)
#         except: 
#             continue
# print(arima)
# print("min aic",min_aic)
# print("optimal arima", min_arima)

mod = sm.tsa.statespace.SARIMAX(y2,order=(1, 1, 1),seasonal_order=(1, 1, 1, 12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(18, 8))
plt.show()

print(type(results))
pred = results.get_prediction(dynamic=False) #start=pd.to_datetime('2020-07-23'), end=pd.to_datetime('2020-07-27')
pred_ci = pred.conf_int()
ax = y['2020':].plot(label='observed')
# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Retail_sold')
plt.legend()
plt.show()