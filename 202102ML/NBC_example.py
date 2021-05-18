from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import f1_score
import NBC
train_ratio=0.8
# load dataset
iris = load_iris()
X, y = iris['data'], iris['target']
N, D = X.shape

#  split train and result sets.
Ntrain = int(train_ratio * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]
temp=NBC.NBC([0,1,2],4)
temp.fit(Xtrain,ytrain)
prd_y=temp.predict(Xtest)
print(f1_score(ytest,prd_y,average = 'macro'))
print("finish")