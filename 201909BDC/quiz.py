dataResult={}
#定义数据转接变量
def emit(tempKey,tempValue):
    if tempKey  not in dataResult:
        dataResult[tempKey]=tempValue
    for key in dataResult:
        for i in range(0,len(dataResult[key])):
            if(dataResult[key][i]==tempKey) :
                tempMap={}
                tempMap[tempKey]=tempValue
                dataResult[key][i]=tempMap
    print(dataResult)

def Map(filename):
    fr=open(filename)
    arrayOflines =fr.readlines()
    #获取行数，即数据总量
    #建立空矩阵
    tempList=[]
    index=0
    for line in arrayOflines:
        if(index==2):
            emit(listFromLine[0], tempList)
            tempList=[]
            index=0
        line=line.strip()
        #截取回车字符
        listFromLine=line.split('\t')
        #利用数字之间的tab拆分成一个数组
        tempList.append(listFromLine[1])
        index=index+1
def reduce(key,values):
    for i in range(0,len(values)):
        tempList = []
        if(isinstance(values[i],dict)):
            for key in values[i]:
                tempList.extend(values[i][key])
        print(key,tempList)


Map("child-parent.txt")
for key in dataResult:
    reduce(key, dataResult[key])