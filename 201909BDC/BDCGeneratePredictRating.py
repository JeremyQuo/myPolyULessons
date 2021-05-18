from sklearn import linear_model
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import matplotlib. pyplot as plt
import math

model = linear_model.LinearRegression()

df=pd.read_csv("training_data.csv",encoding='utf-8',sep=',')
average_rating=df.average_rating
df.drop(['average_rating'], axis=1,inplace=True)

model=model.fit(df, average_rating)

df_test=pd.read_csv("testing_data.csv",encoding='utf-8',sep=',')
prd_y=model.predict(df_test)


predict_f=pd.DataFrame(prd_y)
predict_f.columns=['predict_rating']
predict_f.to_csv('predict_txt.csv',sep=',', header=True, index=True)

