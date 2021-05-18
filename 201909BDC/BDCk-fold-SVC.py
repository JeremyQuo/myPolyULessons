

import sklearn.metrics as sm
import matplotlib. pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.svm import SVR
import pandas as pd
mse=[]
r2=[]

for i in range(0,10):
    #model = RandomForestRegressor()
    # model = GradientBoostingRegressor()
    model =SVR()
    df = pd.read_csv("train_ma"+str(i)+".csv", encoding='utf-8', sep=',')
    average_rating = df.average_rating
    df.drop(['average_rating'], axis=1, inplace=True)
    model = model.fit(df[:9000], average_rating[:9000])
    prd_y = model.predict(df[9000:])
    y=average_rating[9000:]
    mse.append(sm.mean_squared_error(y, prd_y))
    r2.append(sm.r2_score(y, prd_y))
print(np.mean(mse), np.mean(r2))
print(np.std(mse), np.std(r2))