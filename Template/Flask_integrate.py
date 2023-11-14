import pickle
import sklearn.metrics
from flask import Flask, request, render_template
  
   
app=Flask(__name__)
model=pickle.load(open("bestmodel.pkl","rb"))
data_normalizer=pickle.load(open("Scaler.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home_direct():
    return render_template('home.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/output')
def output():
    return render_template("output.html")

@app.route('/submit',methods=['GET','POST'])
def submit():
    Discount_offered=eval(request.form["Discount_offered"])
    Weight_in_gms=eval(request.form["Weight_in_gms"])
    preds=[[Discount_offered,Weight_in_gms]]
    predclass=model.predict(data_normalizer.fit_transform(preds))
    prob=model.predict_proba(data_normalizer.fit_transform(preds))[0]
    reach=prob[0]
    percent=reach*100
    return render_template("output.html", output=percent)

if __name__=='__main__':
    app.run(debug=True)
