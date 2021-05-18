from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd
import sklearn.metrics as sm

df_train=pd.read_csv("train_ma9.csv",encoding='utf-8',sep=',')[:9000]
clf = Ridge(alpha=1)
average_rating=df_train.average_rating
df_train.drop(['average_rating'], axis=1,inplace=True)
clf.fit(df_train, average_rating)

df_test=pd.read_csv("train_ma9.csv",encoding='utf-8',sep=',')[9000:]
y=df_test.average_rating
df_test.drop(['average_rating'], axis=1,inplace=True)
prd_y=clf.predict(df_test)


print("mean_absolute_error:", sm.mean_absolute_error(y, prd_y))
print("mean_squared_error:", sm.mean_squared_error(y, prd_y))
print("median_absolute_error:", sm.median_absolute_error(y, prd_y))
print("R2_score:", sm.r2_score(y, prd_y))
