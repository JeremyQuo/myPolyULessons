from math import log
import operator

#用来对List中元素出现的频率进行排序
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#计算熵
def calculateEntropy(dataSet) :
    numEntries=len(dataSet)
    labelCounts={}
    #对于结构性数据进行分类的数目统计
    for featVec in dataSet :
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    Entropy=0
    for key in labelCounts:
        #本选项被选择的概率
        prob=float(labelCounts[key])/numEntries
        #对于每一项不重复的key进行熵的求和（因为分数进行对数运算之后是负数，所以就是求减）
        Entropy= Entropy-prob*log(prob,2)
    return Entropy
def createDateSet():
    #构建初始数据
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels =['no surfacing','flippers']
    return dataSet,labels

#用以去除每一行中特定列符合要求的值
def splitDataSet(dataSet,axis,value):
    result=[]
    for featVec in dataSet:
        if featVec[axis] ==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            result.append(reducedFeatVec)
    return result

#选择最好的分类方式
def chooseBestFeature2Split(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calculateEntropy(dataSet)
    bestInfoGain=0
    bestFeature=0
    for i in range(numFeatures):
        #分别对每一列进行分类操作之后求信息期望值，
        featList=[example[i] for example in dataSet]
        #取得第i列的数据变成一个list
        uniqueValues=set(featList)
        #去重 得到值域
        newEntropy=0
        for value in uniqueValues:
            subdataSet=splitDataSet(dataSet,i,value)
            prob=len(subdataSet)/float(len(dataSet))
            newEntropy=newEntropy+prob*calculateEntropy(subdataSet)
        #以上为计算信息期望值的公式，即每一个选项被选中的概率乘以熵的总和
        infoGain=baseEntropy-newEntropy
        if(infoGain >bestInfoGain):
            baseEntropy=infoGain
            bestFeature=i
        #以上为判断大小，信息期望值越大，就代表是更好的数据集的划分方式
    return  bestFeature
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #取最后一列为List
    if classList.count(classList[0])==len(classList):
        #如果第一行在数据中一直反复出现（即没有进行再次分类的必要了），则返回第一行
        return  classList[0]
    if len(dataSet[0])==1:
        #用来标识遍历结束时返回最多出现的类别
        return majorityCnt(classList)
    bestFeature=chooseBestFeature2Split(dataSet)
    bestFeatLabel=labels[bestFeature]
    # 获取最高信息期望值的分类位置，和对应的标签
    myTree={bestFeatLabel:{}}
    #得到标签之后构建树
    del(labels[bestFeature])
    #删除已使用的标签
    featValues=[example[bestFeature] for example in dataSet]
    #提出最合适的分类列
    uniqueVals=set(featValues)
    #去重，提取值域
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
        #对于每一个值域中的值，树结构中新建结构
        # 即对于和value匹配的数据，删除已用作分类的bestFeatLabel之后的结构
    return myTree
myData,label=createDateSet()
print(createTree(myData,label))
