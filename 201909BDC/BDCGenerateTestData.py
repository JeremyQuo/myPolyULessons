from sklearn.preprocessing import StandardScaler
import pandas as pd
import category_encoders as encoders
df_train = pd.read_csv("new_Train_data.csv", encoding='utf-8', sep=',')
df_test = pd.read_csv("new_Test_data.csv", encoding='utf-8', sep=',')
target_columns = ['num_pages', 'ratings_count', 'text_reviews_count']
format_train = df_train[target_columns]
format_test = df_test[target_columns]



std = StandardScaler().fit(format_train)
train_standard = std.transform(format_train)
df_train_result = pd.DataFrame(train_standard)

test_standard = std.transform(format_test)
df_test_result=pd.DataFrame(test_standard)
#以上为标准化处理数值型数据


df_train_result.columns=target_columns
df_test_result.columns=target_columns

df_train_rating=df_train['average_rating']

encLea_language = encoders.LeaveOneOutEncoder()
encLea_language.fit(df_train['language_code'],df_train_rating)
df_train['language_code']=encLea_language.transform(df_train['language_code'])
df_test['language_code']=encLea_language.transform(df_test['language_code'])

encLea_publisher = encoders.LeaveOneOutEncoder()
encLea_publisher.fit(df_train['publisher'],df_train_rating)
df_train['publisher']=encLea_publisher.transform(df_train['publisher'])
df_test['publisher']=encLea_publisher.transform(df_test['publisher'])

df_train_result=df_train_result.join(df_train['language_code'])
df_train_result=df_train_result.join(df_train['publisher'])

df_test_result=df_test_result.join(df_test['language_code'])
df_test_result=df_test_result.join(df_test['publisher'])

#以上为利用LeaveOneOutEncoder给出版商和语言种类加权重,并分别给测试集和训练集赋值

authorMap_train={'author1':[],'author2':[],'author3':[]}
count=0
for i in range(0,len(df_train['authors'])):
    line = df_train['authors'][i].split('/')
    tempList=["","",""]
    for j in range(0,len(line)):
        if(j<3):
            tempList[j]=line[j]
        else:
            break
    if(tempList[1]=="" and tempList[2]==""):
        temp = tempList[0]
        tempList[1] = temp
        tempList[2] = temp
    elif(tempList[1]!="" and tempList[2]=="" ):
        tempList[2]=tempList[0]
#        temp = tempList[1]
#        tempList[1] = tempList[0]
#        tempList[2] = temp
    for j in range(0,3):
        authorMap_train['author'+str(j+1)].append(tempList[j])

pd_train_author=pd.DataFrame(authorMap_train)
#以上为提取第一第二第三作者
encLea_author =  encoders.LeaveOneOutEncoder(cols=['author1','author2','author3'])
encLea_author.fit(pd_train_author[['author1','author2','author3']],df_train_rating)
pd_train_author[['author1','author2','author3']]=encLea_author.transform(pd_train_author[['author1','author2','author3']])
df_train_result=df_train_result.join(pd_train_author)


authorMap_test={'author1':[],'author2':[],'author3':[]}
count=0
for i in range(0,len(df_test['authors'])):
    line = df_test['authors'][i].split('/')
    tempList=["","",""]
    for j in range(0,len(line)):
        if(j<3):
            tempList[j]=line[j]
        else:
            break
    if(tempList[1]=="" and tempList[2]==""):
        temp = tempList[0]
        tempList[1] = temp
        tempList[2] = temp
    elif(tempList[1]!="" and tempList[2]=="" ):
        tempList[2]=tempList[0]
#        temp = tempList[1]
#        tempList[1] = tempList[0]
#        tempList[2] = temp
    for j in range(0,3):
        authorMap_test['author'+str(j+1)].append(tempList[j])

pd_test_author=pd.DataFrame(authorMap_test)
pd_test_author[['author1','author2','author3']]=encLea_author.transform(pd_test_author[['author1','author2','author3']])
df_test_result=df_test_result.join(pd_test_author)


df_train_result=df_train_result.join(df_train.average_rating)
df_train_result.to_csv('training_data.csv',sep=',', header=True, index=True)
#model.fit(x_standard, df['average_rating'])
df_test_result.to_csv('testing_data.csv',sep=',', header=True, index=True)