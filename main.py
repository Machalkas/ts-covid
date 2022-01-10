import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import json
warnings.filterwarnings("ignore")


class CovidOracle():
    model,data,columns={},None,None
    __result__,__mod__={},{}
    def __init__(self, data=None, filename:str=None, csv_sep:str=";", df_index:str="Дата", model:tuple=None, load_last_model:bool=False):
        if data:
            self.data=data
        elif filename:
            self.uploadFile(filename, csv_sep, df_index)
        if load_last_model:
            with open("last_model.json","r") as f:
                self.model=json.load(f)
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
    
    def estimateModel(self, ts_columns:list):
        for col in ts_columns:
            aic_full = pd.DataFrame(np.zeros((6,6), dtype=float))
            for p in range(6):
                for q in range(6):
                    if p == 0 and q == 0:
                        continue
                    mod = sm.tsa.statespace.SARIMAX(self.data[col], order=(p,0,q), enforce_invertibility=False)
                    try:
                        res = mod.fit(disp=False)
                        aic_full.iloc[p,q] = res.aic
                    except:
                        aic_full.iloc[p,q] = np.nan
            aic_filter=np.nan_to_num(aic_full)
            p,q=np.where(aic_filter==np.min(aic_filter[np.nonzero(aic_filter)]))
            self.model[col]=(int(p[0]),0,int(q[0]))
            with open("last_model.json","w") as f:
                json.dump(self.model,f)
        self.fitModel(ts_columns)
        return self.model
    
    def fitModel(self, ts_columns:list, model:dict=None):
        if model:
            self.model=model
        for col in ts_columns:
            mod = sm.tsa.statespace.SARIMAX(self.data[col])
            self.__mod__[col]=mod
            self.__result__[col] = mod.fit(disp=False)

    def makePrediction(self, ts_column:str, nforecast):
        predict = self.__result__[ts_column].get_prediction(end=self.__mod__[ts_column].nobs + nforecast)
        predict_ci = predict.conf_int(alpha=0.5)
        return predict.predicted_mean[-nforecast:], predict_ci

# Web API goes here.

import flask

cv = CovidOracle(filename="data/covid.csv", load_last_model=False)
cv.fitModel(cv.columns)

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/columns", methods=["GET"])
def columns():
    return flask.jsonify(cv.columns)

@app.route("/predict", methods=["GET"])
def predict():
    print(cv.columns)
    predictor = flask.request.args.get("predictor")
    numDays = int(flask.request.args.get("numDays"))

    mean, ci = cv.makePrediction(predictor, numDays)
    return flask.jsonify({"mean": mean.to_dict(), "ci": ci.to_dict()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")