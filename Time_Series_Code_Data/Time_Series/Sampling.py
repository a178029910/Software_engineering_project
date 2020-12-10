import numpy as np
import pandas as pd
from Time_Series import Z_normal
import matplotlib.pyplot as plt
def Simple_Sampling(X,n):
    """
    计算简单采样
    """
    X = np.array(X)
    if len(X)<=n:
        return -1
    list=[]
    for i in range(len(X)):
        if i%n == 0:
            list.append(X[i])
    return list
def PAA_Sampling(X,n):
    '''
    PAA计算
    '''
    X = np.array(X)
    if len(X)<=n:
        return -1
    list=[]
    sum=0
    for i in range(len(X)):
        sum+=X[i]
        if i%n == 0 and i != 0:
            sum=sum/n
            list.append(sum)
            sum=0
    return list

def L_Sampling(T,R):
    """
    线性分段计算
    """
    T = np.array(T)
    X = []
    for i in range(0, len(T)):
        X.append((i, T[i]))
    vital_point = []
    vital_point.insert(0, X[0])
    index = 0
    for i in range(1, len(T) - 1):
        if T[i] > T[i - 1] and T[i] > T[i + 1]:
            if T[i] / T[index] > R:
                index += 1
                vital_point.insert(index, X[i])
        if T[i] < T[i - 1] and T[i] < T[i + 1]:
            if T[i] == 0 or T[index] / T[i] > R:
                index += 1
                vital_point.insert(index, X[i])

    index += 1
    vital_point.insert(index, X[len(T) - 1])
    return vital_point




if __name__ == '__main__':
    # data =[2,2,3,4,5,6,7,8,11,21,32,55,11]
    pf=pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\synthetic_data\R15.txt")
    pf1=pf.iloc[:,0].tolist()
    data1=pf1
    # r1 = Simple_Sampling(data1, 5)
    # r2 = PAA_Sampling(data1,5)
    r3 = L_Sampling(data1, 1.05)
    index=[r3[i][0] for i in range(len(r3))]
    zhi=[r3[i][1] for i in range(len(r3))]
    plt.figure(figsize=(12,8),dpi=80)
    plt.plot(range(len(pf1)),pf1,color="b",label="Origin_data")
    # plt.plot(range(len(r1)),r1,color='g',label="Simple_Sampling")
    # plt.plot(range(len(r1),len(r2+r1)), r2, color='r',label="PAA_Sampling")
    plt.plot(index, zhi,  color='r',label="L_Sampling")
    plt.xlabel("time_point",fontsize=12)
    plt.ylabel("data_1D",fontsize=12)
    plt.title("Sampling",fontsize=20)
    plt.legend()
    plt.show()


    # print(PAA_Sampling(data1,5))
    # print(L_Sampling(data1, 2))
    # print(data)
    # df=pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\cnn_data\cifar_features.txt",header=None)
    # print(df)