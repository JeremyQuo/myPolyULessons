import  matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
#定义文本框和箭头格式

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.axl=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=getNumLeafs(inTree)
    plotTree.totalD=getTreeDepth(inTree)
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.axl.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
                            textcoords='axes fraction',va="center",ha="center",
                            bbox=nodeType,arrowprops=arrow_args)
#计算树的叶节点
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict= myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs=numLeafs+getNumLeafs(secondDict[key])
        else:
            numLeafs=numLeafs+1
    return numLeafs
#获取树的层级
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return  maxDepth
#构造初始数据
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]
#在父子节点之间填充文本信息
def plotMidText(cntrPt,parentPt,textString):
    xMid=(parentPt[0]-cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.axl.text(xMid,yMid,textString)

def plotTree(myTree,parentPt,nodeText):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1+numLeafs)/2/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeText)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff + 1/plotTree.totalD


createPlot(retrieveTree(1))



