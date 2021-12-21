print("init...")
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import itertools
print("init done.")

# Register converters to avoid warnings
pd.plotting.register_matplotlib_converters()
plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)

# Dataset
# wpi1 = requests.get('https://www.stata-press.com/data/r12/wpi1.dta').content
# data = pd.read_stata(BytesIO(wpi1))
# print(data)
# data.index = data.t

data=pd.read_csv("data/full_grouped.csv")
data=data.loc[data['Country/Region'] == "Russia"]
data=data.set_index(["Date"])
# data=pd.get_dummies(data, columns=["Date"])
print(data)

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# print(np.asarray(data))
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except ZeroDivisionError: 
            continue
# data.index.freq="QS-OCT"

# # Fit the model
# mod = sm.tsa.statespace.SARIMAX(data['Deaths'], trend='c', order=(1,1,1))
# res = mod.fit(disp=False)
# print(res.summary())

# # Graph data
# fig, axes = plt.subplots(1, 1, figsize=(15,4))

# # Levels
# axes.plot(data.index._mpl_repr(), data['Deaths'], '-')
# axes.set(title='Смертность')

# plt.show()
