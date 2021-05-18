from sklearn import linear_model
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import matplotlib. pyplot as plt
import math

mse=[]
r2=[]
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
columns=["num_pages","ratings_count","text_reviews_count","average_rating"]
for i in range(0,10):
    #model = DecisionTreeRegressor()
    #model = linear_model.Lasso()
    model = linear_model.Ridge()
    df = pd.read_csv("train_ma"+str(i)+".csv", encoding='utf-8', sep=',')
    df=df[columns]
    average_rating = df.average_rating
    df.drop(['average_rating'], axis=1, inplace=True)
    model = model.fit(df[:9000], average_rating[:9000])
    prd_y = model.predict(df[9000:])
    y=average_rating[9000:]
    mse.append(sm.mean_squared_error(y, prd_y))
    r2.append(sm.r2_score(y, prd_y))
print(np.mean(mse),np.mean(r2))
print(np.std(mse),np.std(r2))
