print("init...")
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
from pylab import rcParams
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
print(data)

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
