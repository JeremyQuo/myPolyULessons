
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def load_data():
    (x_train,Y_train), (x_test,Y_test) =tf.keras.datasets.mnist.load_data()
    x_train= x_train.reshape(-1, 28*28).astype("float32") / 255
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255

    return x_train,Y_train, x_test,Y_test

def network(layer_num):
    input_format = tf.keras.Input(shape=784)
    layer_list=[]
    for i in range(layer_num):
        if i==0:
            temp_layer = tf.keras.layers.Dense(256, activation='relu', name='layer'+str(i+1))(input_format)
        else :
            temp_layer = tf.keras.layers.Dense(256, activation='relu', name='layer'+str(i+1))(layer_list[-1])
        layer_list.append(temp_layer)

    # layer1 = tf.keras.layers.Dense(256,activation='relu',name='layer1')(input_format)
    # layer2 = tf.keras.layers.Dense(256, activation='relu', name='layer2')(layer1)
    # layer3 = tf.keras.layers.Dense(256, activation='relu', name='layer3')(layer2)
    # layer4 = tf.keras.layers.Dense(256, activation='relu', name='layer4')(layer3)
    # layer5 = tf.keras.layers.Dense(256, activation='relu', name='layer5')(layer4)

    prd_y = tf.keras.layers.Dense(256, activation='softmax')(layer_list[-1])

    network = tf.keras.Model(inputs=input_format,outputs=prd_y)
    return network

def optimization(network):
    network.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.),
        metrics = ["accuracy"],
    )


def train(x_train, Y_train, net):
    net.fit(x_train,Y_train,batch_size=32,epochs=5)

    return net

def test(x_test,Y_test,net):
    result=net.evaluate(x_test,Y_test,batch_size=32)

    return result[1]

def main():
    x_train, Y_train, x_test, Y_test=load_data()

    train_acc = []
    test_acc = []
    model = network(layer_num=5)
    optimization(model)

    model = train(x_train, Y_train, model)
    test(x_test,Y_test,model)
main()
