from sklearn.preprocessing import StandardScaler
import pandas as pd
import category_encoders as encoders
import numpy as np
#此代码用于将10000条训练集分成测试集和训练集，用以算法自测
df_train = pd.read_csv("new_Train_data.csv", encoding='utf-8', sep=',')

target_columns = ['num_pages', 'ratings_count', 'text_reviews_count']
df_num_train = df_train[target_columns]

interval = 1000
for k in range(0, 10):
    format_train = df_num_train[0:interval * k].append(df_num_train[(k + 1) * interval:])
    format_test = df_num_train[interval * k:interval * (k + 1)]
    std = StandardScaler().fit(format_train)
    format_train = std.transform(format_train)
    format_test = std.transform(format_test)
    temp = np.concatenate((format_train, format_test), axis=0)
    df_train_result = pd.DataFrame(temp)

    # 以上为标准化处理

    df_train_result.columns = target_columns

    df_train_rating = df_train['average_rating'][0:interval * k].append(df_train['average_rating'][(k + 1) * interval:])
    df_test_rating = df_train['average_rating'][interval * k:interval * (k + 1)]

    encLea_language = encoders.LeaveOneOutEncoder()
    language_train_data = df_train['language_code'][0:interval * k].append(df_train['language_code'][(k + 1) * interval:])
    language_text_data = df_train['language_code'][interval * k:interval * (k + 1)]
    encLea_language.fit(language_train_data, df_train_rating)
    language_train_data = encLea_language.transform(language_train_data, df_train_rating).language_code
    language_text_data = encLea_language.transform(language_text_data).language_code

    encLea_publisher = encoders.LeaveOneOutEncoder()

    publisher_train_data = df_train['publisher'][0:interval * k].append(df_train['publisher'][(k + 1) * interval:])
    publisher_text_data = df_train['publisher'][interval * k:interval * (k + 1)]
    encLea_publisher.fit(publisher_train_data, df_train_rating)
    publisher_train_data = encLea_publisher.transform(publisher_train_data, df_train_rating).publisher
    publisher_text_data = encLea_publisher.transform(publisher_text_data).publisher

    df_train_result = df_train_result.join(pd.concat([language_train_data, language_text_data], ignore_index=True))
    df_train_result = df_train_result.join(pd.concat([publisher_train_data, publisher_text_data], ignore_index=True))

    # 以上为利用LeaveOneOutEncoder给出版商和语言种类加权重,并分别给测试集和训练集赋值

    authorMap_train = {'author1': [], 'author2': [], 'author3': []}
    count = 0
    for i in range(0, len(df_train['authors'])):
        line = df_train['authors'][i].split('/')
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
    # 以上为提取第一第二第三作者
    encLea_author = encoders.LeaveOneOutEncoder(cols=['author1', 'author2', 'author3'])

    temp_train = pd_train_author[['author1', 'author2', 'author3']][0:interval * k].append(
        pd_train_author[['author1', 'author2', 'author3']][(k + 1) * interval:])
    temp_test = pd_train_author[['author1', 'author2', 'author3']][interval * k:interval * (k + 1)]
    encLea_author.fit(temp_train, df_train_rating)
    temp_train = encLea_author.transform(temp_train, df_train_rating)
    temp_test = encLea_author.transform(temp_test)
    df_train_result = df_train_result.join(pd.concat([temp_train, temp_test], ignore_index=True))
    df_train_result = df_train_result.join(pd.concat([df_train_rating, df_test_rating], ignore_index=True))

    # pd_train_author[['author1','author2','author3']][:9000]=a[['author1','author2','author3']]
    # pd_train_author[['author1','author2','author3']][9000:]=encLea_author.transform(pd_train_author[['author1','author2','author3']][9000:])[['author1','author2','author3']]

    df_train_result.to_csv('train_ma' + str(k) + '.csv', sep=',', header=True, index=True)
