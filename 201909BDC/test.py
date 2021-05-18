alln=10
#参数:需要凑齐的总金额
coin_list=(1,2,3)
#参数：零钱的数组
def createTable(coin_list,alln):
    matrix = [[0 for i in range(alln+1)] for j in range(len(coin_list))]
    #建立初始数组，动态规划的解决思路就是用这样的数组解决问题
    #先设置所有元素为0，i表示数组中第几位零钱，j表示此时凑齐的金额
    #同时也是为了设置边界
    #matrix[i][j]代表，使用数组前i位的零钱去凑j的金额时，有多少种组合方式
    for i in range(len(coin_list)):
        matrix[i][0]=1
    #初始化1表示，组成0元所用的方式只有一种
    for j in range(alln+1):
        if(j%coin_list[0]==0):
            matrix[0][j]=1
    #初始化2表示，能被总金额整除的单一种类零钱是一种方法
    for i in range(1,len(coin_list)):
        for j in range(1,alln+1):
            #这两个循环的意思是从第一个零钱开始计算有几种解决方式
            #i的循环代表不断地使用新的零钱来组合，来查看有没有新的解决方式
            #j的循环代表使用已知零钱不断地凑新的金额
            #matrix[i][j]代表，使用数组前i位的零钱去凑j的金额时，有多少种组合方式
            #加到最后 matrix[len(coin_list)-1][alln]就是总的解决方式
            #又因为每次累加的是每种零钱的组合方式，无先后之分，所以不存在重复
            if(j<coin_list[i]):
                #判断如果当前凑齐的金额小于当前零钱的话
                #则组成此金额的方式不需要此种零钱，即组成的方式不累加
                #代码表现为 使用i-1所对应的解决方式
                matrix[i][j]=matrix[i-1][j]
            else:
                #否则如果是当前凑齐的金额大于等于当前零钱的话
                #就需要知道加上当前零钱后，会新增多少解决方法
                #
                #j-coin_list[i]是用当前凑成的总额减去当前的零钱所对应的钱a
                #用当前零钱组成钱a的方式就是和之前的零钱组合方式的补集
                #上述原因为我不知道为啥，但应该和最后的数学方程式有关
                #但是我验证了之后发现很对，因为等于零的时候正好是用零钱自己来生成相等的金额
                #
                #因为内嵌的循环是j所以会充满横坐标为i的一维数组
                matrix[i][j]=matrix[i-1][j]+matrix[i][j-coin_list[i]]
    print(matrix[len(coin_list)-1][alln])
createTable(coin_list,alln)
#时间复杂度分析：内嵌一个循环，只有零钱的数组长度和n有关系。
#但总的金额是个常数，所以时间复杂度是O(n)
#同理，所建立的二维数组也是横纵坐标一个是n一个是k，所以空间复杂度是O(n)
#感觉这段代码可以帮忙你证明一下
#第二题我也先不写了 远离算法一年半 公式都不会写了...
#Argue informally that your algorithm is correct, i.e., it counts all ways of making change exactly once, following the restriction described above.
