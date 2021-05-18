import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
import os

# Function to read the file , normalization and split features and labels.
def load_data(data_address,Normalization):
    # read and shuffle the original data
    origin_data = pd.read_csv(data_address)
    origin_data = shuffle(origin_data,random_state=7).reset_index(drop=True)
    # According to the description of data, I selected columns as below which may affect the price
    selected_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF',
                        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'PoolArea', 'SalePrice']
    # delete nan data
    selected_data = origin_data[selected_columns].dropna(axis=0, how='any')
    # Split data to features and labels
    selected_data_price = selected_data['SalePrice']
    selected_data.drop(['SalePrice'], axis=1, inplace=True)
    # Normalization (z-score or MaxminScaler)
    if(Normalization=='z-score'):
        for column_name in selected_data.columns:
            temp_column = selected_data[column_name].values
            temp_column = (temp_column - temp_column.mean()) / temp_column.std()
            selected_data[column_name] = temp_column
    elif(Normalization=='MaxminScaler'):
        for column_name in selected_data.columns:
            temp_column = selected_data[column_name].values
            temp_column = (temp_column - temp_column.min()) / (temp_column.max() - temp_column.min())
            selected_data[column_name] = temp_column
    return selected_data, selected_data_price


# FUnction to add new columns and split data by train_ratio
def polynomial_process(selected_data,selected_data_price,train_ratio,times):

    # add new columns to the represent features of x**times
    data_element = selected_data.values
    result=None
    for i in range(times):
        temp = np.power(data_element, i + 1)
        if result is None:
            result = temp
        else:
            result = np.hstack([result, temp])
    # add column one to represent b
    column_b = np.ones(len(result)).reshape(len(result), -1)
    x = np.hstack([result, column_b])

    # Split data to train and result by train_ratio
    train_num = int(len(x) * train_ratio)
    train_data = x[0:train_num]
    test_data = x[train_num:]
    train_y = selected_data_price[0:train_num]
    test_y = selected_data_price[train_num:]
    return train_data, test_data, train_y, test_y

def gradient_descent(x, y):
    N = x.shape[0]
    column_num = x.shape[1]
    learning_ratio = 0.002
    iterations = 2000
    np.random.seed(2)
    w = np.random.rand(column_num)
    all_avg_err = []
    al_w = [w]
    for i in range(iterations):
        prediction = np.dot(x, w)
        errors = prediction - y.values
        avg_error = 1 / N * np.dot(errors.T, errors)
        all_avg_err.append(avg_error)
        w = w - learning_ratio * (2 / N) * np.dot(x.T, errors)
        al_w.append(w)
    return al_w, all_avg_err


def show_err(all_avg_err):
    plt.title = ("Errors")
    plt.xlabel("No. of iterations")
    plt.ylabel("MSE")
    plt.plot(all_avg_err)
    plt.show()

def show_mse(mse_dict):
    plt.title = ("Errors")
    plt.xlabel("X's times")
    plt.ylabel("MSE")
    for key in mse_dict:
        plt.plot(np.arange(1,len(mse_dict[key])+1,1), mse_dict[key],label=key)
        plt.xticks(range(0,len(mse_dict[key])+1,2))
    plt.legend()
    plt.show()


def calculate_mse(origin_data, origin_y, w):
    prediction = np.dot(origin_data, w)
    mse=mean_squared_error(prediction,origin_y)
    return mse

if __name__ == "__main__":
    origin_data,origin_y = load_data("data/train.csv",Normalization='MaxminScaler')
    mse_list={'train_mse':[],'test_mse':[]}
    for i in range(20):
        train_data, test_data, train_y, test_y = polynomial_process(origin_data,origin_y,0.8, i+1)
        all_w, all_avg_err = gradient_descent(train_data, train_y)
        #show_err(all_avg_err)
        test_mse= calculate_mse(test_data, test_y, all_w[-1])
        train_mse=calculate_mse(train_data, train_y, all_w[-1])
        mse_list['train_mse'].append(train_mse)
        mse_list['test_mse'].append(test_mse)
        print(i)
        #show_dynamic(train_data,train_y,all_w,all_avg_err)
    print(mse_list)
    show_mse(mse_list)

