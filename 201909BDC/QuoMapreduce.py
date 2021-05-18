import time
def computeABS(x,y):
    numberX = x.split('C')[0]
    numberY = y.split('C')[0]
    #转换格式为整型
    numberX=abs(int(numberX)-25)
    numberY=abs(int(numberY)-25)
    if(numberX>numberY):
        return True
    else:
        return False
OutputData={}
#全局变量，用以Map和Reduce之间的数据传输
#同时方便多个Map进行数据的合并
def Map1(filename):
    fr=open(filename)
    arrayOflines =fr.readlines()
    #获取行数，即数据总量
    #建立空矩阵
    index=0
    for line in arrayOflines:
        if index==0:
            index=index+1
            continue
        line=line.strip()
        #截取回车字符
        listFromLine=line.split('\t')
        #利用数字之间的tab拆分成一个数组
        tempMap={}
        tempMap["date"]=listFromLine[1]
        tempMap["temperature"] = listFromLine[3]
        if(listFromLine[0] not in OutputData):
            OutputData[listFromLine[0]]=[tempMap]
        else:
            OutputData[listFromLine[0]].append(tempMap)


def reduce1(key,values):
    minValue = {}
    for item in values:
        #赋予初值
        if(not minValue):
            minValue=item
        elif(computeABS(minValue["temperature"],item["temperature"])):
            #判断哪个离25度更为接近
            minValue = item
    fp = open("result.txt",mode='a')
    print(key,' ',minValue)
    fp.write(minValue["date"]+" is the most pleasant Day in "+key+",and the temperature is "+minValue["temperature"]+".\n")
    fp.close()
    
def main():
    print('This function is build for result')
    print('Please make sure the data.txt is in current folder')
    time.sleep(2)
    print('------------------begin---------------------------')
    print("The output of Map is")
    Map("data.txt")
    print(OutputData)
    fp = open("result.txt", mode='w')
    #Make sure the result.txt is empty to show the correct result
    #But In real Mapreduce,I think it is unnecessary to empty the file cause the mutiple-work.
    #For Mapreduce program running on one PC,the current program is enough for our assignment
    #But when it comes to distributed servers,there is lots of thing can be improved.
    fp.close()
    print("The output of Reduce is")
    for key in OutputData:
        reduce(key,OutputData[key])
    print("And result.txt is generated in current folder")
    print("Program will be closed in 30 seconds")
    time.sleep(30)
    


if __name__ == '__main__':
    main()
    # print(__name__)
