from sklearn import linear_model
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import matplotlib. pyplot as plt
import math
model = linear_model.LinearRegression()
df = pd.read_csv("train_ma9.csv", encoding='utf-8', sep=',')
average_rating = df.average_rating
df.drop(['average_rating'], axis=1, inplace=True)
model = model.fit(df[:9000], average_rating[:9000])
prd_y = model.predict(df[9000:])
y=average_rating[9000:]
y=pd.DataFrame(y).reset_index(drop=True)
y.columns=["y"]
prd_y=pd.DataFrame(prd_y)
prd_y.columns=["prd_y"]
y=y.join(prd_y)
y.to_csv("predict_train_ma9.csv", encoding='utf-8', sep=',')