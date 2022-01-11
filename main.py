import flask
import covidOracle

cv = covidOracle.CovidOracle(filename="data/covid.csv", load_last_model=True)
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