import csv
from datetime import datetime, date
import pandas as pd
import numpy as np

df = pd.read_csv("Train_data.csv", sep=',')
for i in range(0,len(df['publication_date'])):
    line = df['publication_date'][i].split('/')
    df['publication_date'][i]=line[0]
df_new= pd.DataFrame(df['publication_date'])
df_new =df_new.join(df.average_rating)
#df_new.to_csv('time.csv',sep=',', header=True, index=True)
distData={}
for i in range(0,len(df['publication_date'])):
    if df_new['publication_date'][i] not in distData:
        distData[df_new['publication_date'][i]]=[]
    distData[df_new['publication_date'][i]].append(df_new['average_rating'][i])
distData_final={'month':[],'mean':[],'std':[]}
for item in distData:
    print(item,len(distData[item]))
    std=np.std(distData[item])
    mean=np.mean(distData[item])
    distData_final['month'].append(item)
    distData_final['mean'].append(mean)
    distData_final['std'].append(std)
df_new1=pd.DataFrame(distData_final)
df_new1.to_csv('time.csv',sep=',', header=True, index=True)
