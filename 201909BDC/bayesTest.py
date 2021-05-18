from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)
def setOfWord2Vec(vocabList,inputSet):
    returnVec =[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
        else:
            print("单词",word,"不在我的词汇表里")
    return  returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDoc=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/numTrainDoc
    p0Num=zeros(numWords)
    p1Num=zeros(numWords)
    p0Denom=0
    p1Denom=0
    for i in range(numTrainDoc):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec=p1Num/p1Denom
    p0Vec=p0Num/p0Denom
    return p0Vec,p1Vec,pAbusive


listOPost,listClasses=loadDataSet()
myVocabList=createVocabList(listOPost)
trainMat=[]
for postinDoc in listOPost:
    trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
pV,p1V,pAb=trainNB0(trainMat,listClasses)
print(pAb)
# print(setOfWord2Vec(myVocabList,listOPost[0]))