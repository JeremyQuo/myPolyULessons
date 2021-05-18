import csv
from datetime import datetime, date
import pandas as pd

totalData={}

def emit(filename,df):
    global totalData
    if filename not in totalData:
        totalData[filename]=df
    else:
        totalData[filename]=totalData[filename].append(df)
def Func(fileName):
    df = pd.read_csv(fileName, sep=',')
    target_columns=['title', 'authors', 'average_rating','language_code', 'num_pages',
                  'ratings_count', 'text_reviews_count', 'publication_date', 'publisher']
    df=df[target_columns]
    df['author_num']=[None]*len(df['publication_date'])
    df['publisher_num'] = [None] * len(df['publication_date'])
    z = 0
    for i in range(0,len(df['publication_date'])):

        try:
            utcTime1 = datetime.strptime(df['publication_date'][i], "%M/%d/%Y")
            utcTime2 = datetime.strptime("1970-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
            metTime = utcTime1 - utcTime2  # 两个日期的 时间差
            timeStamp = metTime.days * 24 * 3600 + metTime.seconds
            df['publication_date'][i]=timeStamp

            num_author=df['authors'][i].count('/')
            df['author_num'][i]=num_author+1
            num_publisher = df['publisher'][i].count('/')
            df['publisher_num'][i] = num_publisher + 1
            #作者个数计数操作
        except:
            #print(i)
            df.drop(i,axis = 0,inplace=True)
            i=i-1
            z=z+1
            continue
    print(z)
    emit(fileName,df)
def SaveFunc(key,value):
    value.to_csv('new_'+key,sep=',', header=True, index=True)
Func("Train_data.csv")
for key in totalData:
    SaveFunc(key,totalData[key])
