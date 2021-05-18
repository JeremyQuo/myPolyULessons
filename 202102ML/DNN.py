
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def load_data():
    (x_train,Y_train), (x_test,Y_test) =tf.keras.datasets.mnist.load_data()
    x_train= x_train.reshape(-1, 28*28).astype("float32") / 255
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255

    return x_train,Y_train, x_test,Y_test

def network(layer_num,via_num,hidden_activation,output_activation,rate):
    input_format = tf.keras.Input(shape=784)
    layer_list=[]
    for i in range(layer_num):
        if i==0:
            temp_layer = tf.keras.layers.Dense(via_num, activation=hidden_activation, name='layer'+str(i+1))(input_format)
        else:
            temp_layer = tf.keras.layers.Dense(via_num, activation=hidden_activation, name='layer'+str(i+1))(layer_list[-1])
        layer_list.append(temp_layer)

    temp_layer = tf.keras.layers.Dropout( rate=rate,name='layer' + str(6))(layer_list[-1])

    prd_y = tf.keras.layers.Dense(256, activation=output_activation)(temp_layer)

    network = tf.keras.Model(inputs=input_format,outputs=prd_y)
    return network

def optimization(network,loss_function,momentum,learning_rate):
    network.compile(
        loss=loss_function,
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),
        metrics = ["accuracy"],

    )


def train(x_train, Y_train, net,batch_size,epochs):
    net.fit(x_train,Y_train,batch_size=batch_size,epochs=epochs)

    return net

def test(x_test,Y_test,net,batch_size):
    result=net.evaluate(x_test,Y_test,batch_size=batch_size)

    return result[1]

def main():
    x_train, Y_train, x_test, Y_test=load_data()

    train_acc = []
    test_acc = []
    via_list=[0, 0.2, 0.4, 0.6, 0.8, 0.999999]
    via_list_name=[0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in range(len(via_list)):
        model=network(layer_num=5,via_num=256,hidden_activation='relu',output_activation='softmax',rate=via_list[i])
        optimization(model,loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),momentum=0., learning_rate=0.001)

        model=train(x_train,Y_train,model,batch_size=32,epochs=5)
        history=model.history
        train_acc.append(history.history['accuracy'][-1])
        test_acc.append(test(x_test, Y_test,model,batch_size=32))
    plt.title('Result Analysis')
    plt.plot(via_list_name, train_acc, color='green', label='training accuracy')
    plt.plot(via_list_name, test_acc, color='red', label='testing accuracy')
    plt.legend()
    plt.show()


main()
