# Mark Evers
# Created: 4/9/2018
# webapp.py
# Web application


import flask
import pickle
from model_final import load_from_disk


app = flask.Flask(__name__)
# model = load_from_disk("models/final")




@app.route("/", methods=['GET'])
def index():
    return flask.render_template("shell.html")

@app.route("/submit", methods=['GET'])
def submit():
    return flask.render_template("submit.html")
#
# @app.route("/predict", methods=['POST'])
# def predict():
#     text = flask.request.form["article_text"]
#     prediction = model.predict_one(text)
#
#     return flask.render_template("predict.html", article_section=prediction)
#
#


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
