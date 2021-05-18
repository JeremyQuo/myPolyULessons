def bag(c, w, v):
    value = [[0 for j in range(c + 1)] for i in range(len(w) + 1)]
    #建立动态规划的二维数组
    for i in range(1, len(w) + 1):
        for j in range(1, c + 1):
            #i代表每一个公司
            #j代表第j分钟
            value[i][j] = value[i - 1][j]
            #如果不计算当前公司，最大得分为value[i - 1][j]
            if j >= w[i - 1] and value[i][j] < value[i - 1][j - w[i - 1]] + v[i - 1]:
                # 判断当前时间是否支持当前公司面试时间
                # 根据公式进行比较和替换
                value[i][j] = value[i - 1][j - w[i - 1]] + v[i - 1]
    print(value[len(w)][c])
    print(value);
    return value
timeList=[20,100,5,60,30,170,90,15]
scoreList=[90,90,5,60,55,71,44,95]
totalTime=360
bag(totalTime,timeList,scoreList)
