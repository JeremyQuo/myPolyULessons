import numpy as np

#Function i
def cost(a, y):
    total = 0
    for i in range(0, 2):
        total = total + 0.5 * ((a[i] - y[i]) ** 2)
    print(total)

#初始值
theta1 = np.array([[0.3, 0.9, 1], [0.6, 0.8, -0.3]]);
theta2 = np.array([[0.3, 0.8, 0.2], [-0.1, 0, -0.6]]);
x = np.array([1, 0.9, 0.1])
y = np.array([0, 1])

#点乘（负数取0）
a2 = np.dot(theta1, x)
a2 = np.insert(a2, 0, 1)
a3 = np.dot(theta2, a2)
a3[a3 < 0] = 0
cost(a3, y)

for i in range(0, len(theta1)):
    for j in range(0, len(theta1[i])):
        temp1 = a3[0] - y[0]
        temp2 = a3[1] - y[1]
        if(a3[0]==0):
            temp1=0
        if (a3[1] == 0):
            temp2=0
        temp = temp1 * theta2[0][i+1] + temp2 * theta2[1][i+1]
        #本图的向后传播theta1和a(2)的bias无关
        theta1[i][j] = theta1[i][j] - 0.01 * temp * x[j]
print(theta1)

for i in range(0, len(theta2)):
    for j in range(0, len(theta2[i])):
        temp1 = a3[i] - y[i]
        if a3[i]==0:
            temp1=0
        temp = 0.01 * a2[j] * temp1
        theta2[i][j] = theta2[i][j] - temp
print(theta2)

#计算第一次迭代之后的function i 值
a2 = np.dot(theta1, x)
a2[a2 < 0] = 0
a2 = np.insert(a2, 0, 1)
a3 = np.dot(theta2, a2)
a3[a3 < 0] = 0
cost(a3, y)
