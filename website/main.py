from flask import Flask,render_template,request
app = Flask(__name__)
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import svm


#file = open('model.pkl','rb')
#clf = pickle.load(file)
#file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    
    if request.method == "POST":
        MyDict = request.form
        text = (MyDict['text'])
        
        label = model.predict(tfidf.transform([text]))[0]
        
        print(label)
        return render_template('show.html',label=label)
 
    return render_template('index.html')

if __name__ == "__main__"    :
        
    df = pd.read_csv('D:\cyberbullycode\public_data_labeled.csv')
    df_x = df["Text"]
    df_y = df["label"]
    x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size = 0.2,random_state = 4)
    tfidf = TfidfVectorizer()
    x_traincv = tfidf.fit_transform(x_train)
    x_testcv = tfidf.transform(x_test)
    model = svm.SVC()
    model.fit(x_traincv,y_train)
    app.run(debug = True)    