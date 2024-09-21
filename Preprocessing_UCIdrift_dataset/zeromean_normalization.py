import numpy as np
from sklearn.preprocessing import StandardScaler  # 均值方差归一化

# 遍历 batch1.dat 到 batch10.dat
for batch_num in range(1, 11):
    # 打开相应的 batch 文件
    filename = f'batch{batch_num}.dat'
    output_filename = f'../normalization/zeromean_ normalization/batch{batch_num}_ext.txt'

    with open(filename) as fp:
        lines = fp.readlines()  # 读取文件内容

    # 初始化变量
    i = 0
    a = [[0] * 129 for k in range(len(lines))]
    b = [[0] * 128 for z in range(len(lines))]

    for line in lines:
        t1 = 1  # 0 为 str;  1 为space或者tab
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

    # 提取数值
    for i in range(len(a)):
        for j in range(len(a[0]) - 1):
            b[i][j] = float(a[i][j + 1][a[i][j + 1].find(':') + 1:])

    # 转换为 numpy 数组
    X = np.array(b)
    X = X.reshape(len(lines), 128)

    # 标准化
    std_scaler = StandardScaler()
    std_scaler.fit(X)  # 计算均值和方差
    X_std = std_scaler.transform(X)  # 数组标准化

    # 保存标准化后的数组到文件
    np.savetxt(output_filename, X_std, fmt='%f', delimiter=' ')

    print(f'Processed and saved: {output_filename}')

