import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 最值归一化

for batch_num in range(1, 11):  # 从 batch1.dat 到 batch10.dat
    i = 0
    data = []
    t = 0
    separator = ':'
    B = []

    # 打开对应的 batch 文件
    fp = open(f'batch{batch_num}.dat')  # 动态生成文件名
    lines = fp.readlines()  # 类型为 list
    fp.close()

    # 初始化 a 和 b 数组
    a = [[0] * 129 for k in range(len(lines))]
    b = [[0] * 128 for z in range(len(lines))]

    for line in lines:
        t1 = 1  # 0 为 str; 1 为 space 或者 tab
        t2 = 1
        j = 0
        for x in line:
            if t1 == 0 and t2 == 1:  # 01 str-空
                j += 1
            t1 = t2
            if x != '	' and x != ' ':
                t2 = 0
            else:
                t2 = 1
            if t1 == 1 and t2 == 0:  # 10 空-str
                a[i][j] = x
            else:
                if t1 == 0 and t2 == 0:  # 00 str-str
                    a[i][j] += x
        i += 1

    # 提取数据并转换为浮点型
    for i in range(len(a)):
        for j in range(len(a[0]) - 1):
            b[i][j] = float(a[i][j + 1][a[i][j + 1].find(':') + 1:])

    # 转换为 numpy 数组并进行最值归一化
    X = np.array(b)
    X = X.reshape(len(lines), 128)

    # 归一化处理
    std_scaler = MinMaxScaler()
    std_scaler.fit(X)  # 计算最值
    X_std = std_scaler.transform(X)  # 归一化处理

    # 保存归一化结果到文件
    np.savetxt(f"../normalization/maxmin_Normalization/batch{batch_num}_nor.txt", X_std, fmt='%f', delimiter=' ')

    print(f"batch{batch_num}.dat 处理完成，保存到 batch{batch_num}_nor.txt")

#for i in range (len(a)):
#    print(a[i])









