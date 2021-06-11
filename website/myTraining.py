import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

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

    file = open('model.pkl','wb')
    pickle.dump(model,file)
    file.close()
