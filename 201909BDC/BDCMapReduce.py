from datetime import datetime, date
import pandas as pd
import numpy as np

totalData={}
outputFile = r'BDC_date_analysis.csv'
def emit(key,value):
    global totalData
    if key not in totalData:
        totalData[key]=[]
    totalData[key].append(value)
def Map(fileName):
    df = pd.read_csv(fileName, sep=',')
    publication_date=df['publication_date']
    average_rating=df['average_rating']
    for i in range(0,len(publication_date)):
        line = publication_date[i].split('/')
        month=line[0]
        year=line[1]
        emit(month,average_rating[i])
        # emit(year,average_rating[i])
        # 如果要计算年份的话，把key改一下就好
def Reduce(key,valueList):
    #Reduce用来计算每一类的平均值和标准差
    mean=np.mean(valueList)
    std=np.std(valueList, ddof=1)
    with open(outputFile, 'a') as f:
        f.write(str(key)+','+str(mean)+','+str(std) + '\n')
Map('Train_data.csv')
for key in totalData:
    Reduce(key,totalData[key])