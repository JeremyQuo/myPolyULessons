import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score


class NBC:
    # implement a Na¨ıve Bayes Classifier directly in python
    avg_feature = {}
    sig_feature = {}
    feature_types = None
    num_classes = 1
    prior_class_probability = {}

    # initialization
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes

    # compute normal distribution in log space
    def prob(self, data, avg, std):

        sqrt_2pi = np.power(2 * np.pi, 0.5)
        coef = 1 / (sqrt_2pi * std)
        powercoef = -1 / (2 * np.power(std, 2))
        mypow = powercoef * (np.power((data - avg), 2))

        return np.log(coef * (np.exp(mypow)))

    # find and record the current max posterior_class_probability and return its index
    def find_max(self,matrix_a,matrix_b):
        if(len(matrix_a)==0):
            return matrix_b, None
        index = np.argwhere(matrix_b > matrix_a)
        matrix_a[index]=matrix_b[index]
        return matrix_a, index

    def fit(self, train_data, train_y):

        # for item in self.feature_types:
        #     self.avg_feature[item] = []
        #     self.sig_feature[item] = []

        for item in self.feature_types:
            #  compute π c
            num_c = Counter(train_y)[item]
            self.prior_class_probability[item] = num_c / len(train_y)

            #  record all conditionally distributions of all features(mean/std).
            index = np.argwhere(train_y == item)
            item_feature = train_data[index]
            self.avg_feature[item] = np.mean(item_feature, axis=0)[0]
            self.sig_feature[item] = np.std(item_feature, axis=0)[0]
        #print("finished training")

    def predict(self, test_data):

        # resultList = []
        if (len(self.avg_feature) == 0):
            print("please run fit first")

        # obtain the prediction of all Xtest points

        result_list = np.full((len(test_data)), self.feature_types[0]).astype(int)
        feature_list = []
        for item in self.feature_types:
            test = self.prob(test_data, self.avg_feature[item], self.sig_feature[item])
            # ln(ab)=lna+lnb ,so addition will replace multiplication
            test_sum = np.sum(test, axis=1)
            posterior_class_probability = self.prior_class_probability[item] * test_sum
            # get the max posterior_class_probability of ci and return its index
            feature_list, index = self.find_max(feature_list, posterior_class_probability)
            result_list[index] = item
        return result_list

        # the original code used lots of loops
        # for row in test_data:
        #     probList = {}
        #     for item in self.feature_types:
        #         theta=self.prob(row, self.avg_feature[item], self.sig_feature[item])
        #         # ln(ab)=lna+lnb ,so addition will replace multiplication
        #         temp = np.sum(theta)
        #         posterior_class_probability = self.prior_class_probability[item]* temp
        #         probList[item] = posterior_class_probability
        #     resultList.append(max(probList, key=probList.get))
        # return resultList


def example(train_ratio=0.8):

    # load dataset
    iris = load_iris()
    X, y = iris['data'], iris['target']
    N, D = X.shape
    f1_list=[]
    acc_list=[]

    # to ensure the result is stable
    np.random.seed(7)

    for i in range(20):
        #  split train and result sets.
        Ntrain = int(train_ratio * N)
        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        Xtest = X[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]

        #  build and use NBC classifier
        temp = NBC(feature_types=[0, 1, 2], num_classes=4)
        temp.fit(Xtrain, ytrain)
        prd_y = temp.predict(Xtest)
        f1_list.append(f1_score(ytest, prd_y, average='macro'))
        acc_list.append(np.mean(prd_y == ytest))

    # compute the final training and testing accuracy
    print("The value of macro_f1 is",np.mean(f1_list))
    print("The value of accuracy is",np.mean(acc_list))

if __name__ == "__main__":
    # execute only if run as a script
    example()
