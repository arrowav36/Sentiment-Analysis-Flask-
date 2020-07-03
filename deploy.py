from flask import Flask, render_template, request
import numpy as np
import sentiment as S

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def home():
    data = request.form['review']
    reviews_arr=np.array([data])
    reviews_vector=S.vectorizer.transform(reviews_arr) 
    Ans = S.clf.predict(reviews_vector)
    return render_template('result.html', data=Ans)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__=="__main__":
    app.run(debug=True)



