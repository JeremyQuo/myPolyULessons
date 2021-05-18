from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plot
import nodeClassifier as nC
def file2matrix(filename):
    fr=open(filename)
    arrayOflines =fr.readlines()
    numerOfLines=len(arrayOflines)
    #获取行数，即数据总量
    returnMat=zeros((numerOfLines,3))
    #建立空矩阵
    classLabelVector=[]
    index=0
    for line in arrayOflines:
        line=line.strip()
        #截取回车字符
        listFromLine=line.split('\t')
        #利用数字之间的tab拆分成一个数组
        returnMat[index:]=listFromLine[0:3]
        #前三个数录入特征矩阵
        classLabelVector.append(int(listFromLine[-1]))
        #最后一个数作为分类向量
        index=index+1
    return returnMat,classLabelVector
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    #选出每一列的最小值组成的一维向量
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet - tile(minVals,(m,1))
    #每一个值减最小值
    normDataSet= normDataSet/tile(ranges,(m,1))
    #除以最大值和最小值的差
    return normDataSet,ranges,minVals


def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normDataMat, ranges, minVals = autoNorm(datingDataMat)
    m=normDataMat.shape[0]
    numTestVecs=int(m*hoRatio)
    #取百分之十作为测试，百分之九十用来训练
    errorCount=0
    for i in range(numTestVecs):
        classifierResult = nC.classfy0(normDataMat[i,:],normDataMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        dict0={1:'不满意',2:'满意',3:'非常满意'}
        print("分类器返回结果为",dict0.get(classifierResult),"实际结果为",dict0.get(datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount=errorCount+1
    print("错误率为",errorCount/numTestVecs)
datingClassTest()