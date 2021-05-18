from numpy import *
import operator
import array
def createDataSet():
    group=array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#group,labels=createDataSet()
def classfy0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #使得矩阵大小相同，从而做减法运算
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    #axis=1代表每一行之前求和，为0表示每一列之间求和
    distances=sqDistances**0.5
    #以上为计算目标矩阵和训练集的距离,即相减之后 求平方的和，最后开方
    sortedDistanceRank=distances.argsort()
    #argsort排序，首先按照从小到大排序，之后将值变为之前位置的标号
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistanceRank[i]]
        #argsort的作用下 正好对应到之前的位置，所以找到对应的分类标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount= sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #从根据频率大到小的排名
    return sortedClassCount[0][0]
#print(classfy0([0,0],group,labels,3))