import os
from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pickle
import sklearn
from sklearn.preprocessing  import StandardScaler

app = Flask(__name__)


@app.route('/',methods=['GET'])
@cross_origin()
def home_page():
    return render_template("home.html")


@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():
    x = request.form
    try:
        crim = float(request.form['CRIM'])
        zn = float(request.form['ZN'])
        indus = float(request.form['INDUS'])
        chas = float(request.form['CHAS'])
        nox = float(request.form['NOX'])
        rm = float(request.form['RM'])
        age = float(request.form['AGE'])
        dis = float(request.form['DIS'])
        rad = float(request.form['RAD'])
        ptratio = float(request.form['PTRATIO'])
        b = float(request.form['B'])
        lstat = float(request.form['LSTAT'])
    except:
        return render_template("predict.html",prediction="",head="Please go back and enter correct values in all inputs!")

    filename = 'finalized_model.pickle'
    loaded_model = pickle.load(open(filename, 'rb'))
    input = [[crim, zn, indus, chas, nox, rm, age,dis,rad,ptratio,b,lstat]]
    sc = StandardScaler()
    input_scaled = sc.fit_transform(input)
    prediction = loaded_model.predict(input_scaled)
    print('prediction is', prediction)
    return render_template("predict.html",head="Your Median House Value is!!",prediction=prediction[0])








port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(host='0.0.0.0', port=port)


