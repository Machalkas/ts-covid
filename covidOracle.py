from typing_extensions import Self
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from scipy.stats import norm
from scipy.stats.stats import mode
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import warnings
import json
warnings.filterwarnings("ignore")


class CovidOracle():
    model,data,columns=None,None,None
    __result__,__mod__=None,None
    def __init__(self, data=None, filename:str=None, csv_sep:str=";", df_index:str="Дата", model:tuple=None, load_last_model:bool=False):
        if data:
            self.data=data
        elif filename:
            self.uploadFile(filename, csv_sep, df_index)
        if load_last_model:
            with open("last_model.json","r") as f:
                self.model=tuple(json.load(f))
            self.fitModel()
        elif model:
            self.model=model

    def uploadFile(self, fn:str=None, separator=";", index:str="Дата"):
        dt=pd.read_csv(fn, sep=separator)
        dt=dt.loc[dt['Страна'] == "Россия"]
        dt.drop("Страна",1, inplace=True)
        # dt=dt.loc[:,["Дата","Заражений за день"]]
        dt=dt.set_index([index])
        self.columns=list(dt.columns)
        self.data=dt
    
    def estimateModel(self):
        aic_full = pd.DataFrame(np.zeros((6,6), dtype=float))
        for p in range(6):
            for q in range(6):
                if p == 0 and q == 0:
                    continue
                mod = sm.tsa.statespace.SARIMAX(self.data["Заражений за день"], order=(p,0,q), enforce_invertibility=False)
                try:
                    res = mod.fit(disp=False)
                    aic_full.iloc[p,q] = res.aic
                except:
                    aic_full.iloc[p,q] = np.nan
        aic_filter=np.nan_to_num(aic_full)
        p,q=np.where(aic_filter==np.min(aic_filter[np.nonzero(aic_filter)]))
        self.model=(int(p[0]),0,int(q[0]))
        with open("last_model.json","w") as f:
            json.dump(self.model,f)
        self.fitModel()
        return self.model
    
    def fitModel(self, model:tuple=None):
        if model:
            self.model=model
        mod = sm.tsa.statespace.SARIMAX(self.data["Заражений за день"], order=self.model)
        self.__mod__=mod
        self.__result__ = mod.fit(disp=False)

    def makePrediction(self, nforecast):
        predict = self.__result__.get_prediction(end=self.__mod__.nobs + nforecast)
        predict_ci = predict.conf_int(alpha=0.5)
        return predict.predicted_mean[-nforecast:], predict_ci

        


# cv=CovidOracle(filename="data/covid.csv")
# # print("Estimating...")
# print("Model:",cv.estimateModel())
# print("Prediction:",cv.makePrediction(10)[0])
# # print(type(cv.data))