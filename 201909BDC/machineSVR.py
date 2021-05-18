import pandas as pd
from sklearn.svm import LinearSVR
import  datetime
import sklearn.metrics as sm

dtr=LinearSVR()

df=pd.read_csv("training_data1.csv",encoding='utf-8',sep=',')[:9000]
average_rating=df.average_rating
df.drop(['average_rating'], axis=1,inplace=True)

dtr=dtr.fit(df, average_rating)

df_test=pd.read_csv("training_data1.csv",encoding='utf-8',sep=',')[9000:]
y=df_test.average_rating
df_test.drop(['average_rating'], axis=1,inplace=True)
prd_y=dtr.predict(df_test)

print("mean_absolute_error:", sm.mean_absolute_error(y, prd_y))
print("mean_squared_error:", sm.mean_squared_error(y, prd_y))
print("median_absolute_error:", sm.median_absolute_error(y, prd_y))
print("R2_score:", sm.r2_score(y, prd_y))