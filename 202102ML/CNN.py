import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import heapq


# use cuda or not
use_cuda = torch.cuda.is_available()
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# prepare data
def load_data():
    (x_train, Y_train), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28).astype("float32") / 255
    return x_train, Y_train, x_test, Y_test

# random choose 1 image in each class
# return a dict
def collectNShuffle(x, y):
    # classify
    dict_list = {}
    for i in range(len(x)):
        img = x[i]
        label = y[i]
        if label not in dict_list:
            dict_list[label] = []
        dict_list[label].append(img)
    # random choose one img
    for key in dict_list:
        random_int = random.randint(0, len(dict_list[key]))
        dict_list[key] = dict_list[key][random_int]
    return dict_list

# generate pictures according the dict from collectNShuffle function
def generatePic(x, y, name):
    dict_list = collectNShuffle(x, y)
    for key in dict_list:
        image = dict_list[key]
        plt.subplot(3, 4, key + 1)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title("Truth: {}".format(key))
    plt.savefig('./' + name + '.jpg')
    plt.show()

def task1():
    x_train,Y_train, x_test,Y_test=load_data()
    # Task 1
    # download the MNIST dataset, and randomly visualise 1 image for each class
    # in both training and testing sets. In total, 20 images should be visualised. (10 marks)
    generatePic(x_train,Y_train,'train')
    generatePic(x_test,Y_test,'test')

# Task 2

#  1. build a network with the following layers
#  reshape them as 28x28x1 images (1 because these are grey scale);

# x_train,Y_train, x_test,Y_test=load_data()


def network():
    input_format = tf.keras.Input(shape=(28, 28, 1))

    # Add a convolutional layer with 25 filters of size 12x12x1 and the ReLU non-linearity.
    # Use a stride of 2 in both directions and ensure that there is no padding;
    layer0 = tf.keras.layers.Conv2D(filters=25, kernel_size=(12, 12), strides=(2, 2), activation='relu', name='layer0',
                                    padding="valid")(input_format)

    # Add a second convolutional layer with 64 filters of size 5x5x25 that maintains the same
    # width and height. Use stride of 1 in both directions and add padding as necessary and
    # use the ReLU non-linearity.
    layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='layer1',
                                    padding="same")(layer0)
    # Add a max pooling layer with pool size 2x2
    temp_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(layer1)

    flatten = tf.keras.layers.Flatten()(temp_pool)

    # Add a fully connected layer with 1024 units. Each unit in the max pool should be
    # connected to these 1024 units. Add the ReLU non-linearity to these units
    layer2 = tf.keras.layers.Dense(units=1024, activation='relu', name='layer2')(flatten)

    # Add another fully connected layer to get 10 output units. Don’t add any non-linearity
    # to this layer, we’ll implement the softmax non-linearity as part of the loss function.
    output_layer = tf.keras.layers.Dense(units=10, name='output_layer')(layer2)

    network = tf.keras.Model(inputs=input_format, outputs=output_layer)
    return network


def optimization(network):
    network.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )


def train(x_train, Y_train, net, batch_size=50, epochs=10):
    net.fit(x_train, Y_train, batch_size=batch_size, epochs=epochs,steps_per_epoch=100)

    return net

def test(x_test, Y_test, net):
    result = net.evaluate(x_test, Y_test)
    return result[1]

# visualise the filters in the first convolutional layer
def visualise_filters(model):
    filters=model.layers[1].weights[0].numpy()
    filters=filters.reshape(12,12,25)
    for i in range(25):
        image = filters[:,:,i]
        plt.subplot(5, 5, i + 1)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title(" {}".format(i+1))
    plt.savefig('./filtersOf1stConv.jpg')
    plt.show()
    plt.close()

# to obtain the patch in 1 picture according to the index
def retrieve_patch(pic,x,y):
    return pic[x:x+12,y:y+12]

# visualize the top 12 of one filter
def visualise_patches(result_data):
    for i in range(len(result_data)):
        for j in range(len(result_data[i])):
            image = result_data[i,j,:,:]
            plt.subplot(5, 3, j + 1)
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image, cmap='gray', interpolation='none')
            plt.title("Top {}".format(j+1))
        plt.savefig('./Top12Patches_filter{}.jpg'.format(i+1))
        # plt.show()


#  visualise the patches in the test dataset (10000 images) that result
#  in the highest activation for each filter
def print_conv(net,x_test):
    # construct the sub-model
    sub_model =  tf.keras.Model(inputs = net.input,outputs = net.layers[1].output)
    sub_result=sub_model.predict(x_test)
    sub_result=sub_result.reshape(810000, 25)
    top12_index=[]

    # pick the top 12
    for i in range(25):
        temp_data=sub_result[:,i]
        top12_index.append(np.argpartition(temp_data, -12)[-12:])

    # transfer the index to patches
    result_data=np.zeros(shape=(25,12,12,12))
    for i in range(len(top12_index)):
        for j in range(len(top12_index[i])):
            temp=top12_index[i][j]
            pic_num=temp //(9*9)
            col_num=(temp-pic_num*81)//9
            row_num=(temp-pic_num*81)%9
            temp_pic = retrieve_patch(x_test[pic_num],col_num, row_num)
            result_data[i][j] = temp_pic
    #visualize
    visualise_patches(result_data)

def main():

    x_train, Y_train, x_test, Y_test = load_data()

    # use OneHotEncoder to process Y
    enc = OneHotEncoder()
    Y_train = Y_train.reshape(-1, 1)
    Y_train = enc.fit_transform(Y_train).toarray()
    Y_test = Y_test.reshape(-1, 1)
    Y_test = enc.fit_transform(Y_test).toarray()

    # build a network as required
    model = network()
    print(model.summary())
    optimization(model)
    # Training
    # using minibatches of size 50. Try about 1000-5000 iterations.
    model = train(x_train, Y_train, model,epochs=100)
    prd_y = test(x_test, Y_test, model)

    # Visualising Filters
    visualise_filters(model)
    #  Visualising Patches with High Activation
    print_conv(model,x_test)

if __name__ == '__main__':
    task1()
    main()
print(1)


