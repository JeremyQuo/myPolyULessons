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
    target_columns=[ 'authors', 'average_rating','language_code', 'num_pages',
                  'ratings_count', 'text_reviews_count', 'publisher']
    df=df[target_columns]
    authorMap_train = {'author1': [], 'author2': [], 'author3': []}
    for i in range(0, len(df['authors'])):
        line = df['authors'][i].split('/')
        tempList = ["", "", ""]
        for j in range(0, len(line)):
            if (j < 3):
                tempList[j] = line[j]
            else:
                break
        if (tempList[1] == "" and tempList[2] == ""):
            temp = tempList[0]
            tempList[1] = temp
            tempList[2] = temp
        elif (tempList[1] != "" and tempList[2] == ""):
            tempList[2] = tempList[0]
        #        temp = tempList[1]
        #        tempList[1] = tempList[0]
        #        tempList[2] = temp
        for j in range(0, 3):
            authorMap_train['author' + str(j + 1)].append(tempList[j])

    pd_train_author = pd.DataFrame(authorMap_train)
    df=df.join(pd_train_author)
    df.drop(['authors'], axis=1, inplace=True)
    emit(fileName,df)
    df.to_csv('new_try_'+fileName,sep=',', header=True, index=True)
Func("Train_data.csv")

