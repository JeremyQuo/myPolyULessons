from auto_ml import Predictor
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import matplotlib. pyplot as plt
import math

df = pd.read_csv("train_ma0.csv", encoding='utf-8', sep=',')
average_rating = df.average_rating
# df.drop(['average_rating'], axis=1, inplace=True)

column_descriptions = {
'average_rating':'output',
}
ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df[:9000])

ml_predictor.score(df[9000:],df[9000:].average_rating)