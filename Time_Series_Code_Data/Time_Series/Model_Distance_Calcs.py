import time

import numpy as np
import pandas as pd
from Time_Series.Z_normal import Z_normal


def Calcs_Euclidean_distance(X,Y):
    X=np.array(X)
    Y=np.array(Y)
    if len(X)<=0| len(X)<=0:
        return -1
    if len(X) == len(Y):
        X = Z_normal(X)
        Y = Z_normal(Y)
        dist = np.cumsum([pow(X[i] - Y[i], 2) for i in range(len(X))])
        # print(dist)
        distance = pow(float(dist[len(X)-1]) / len(X), 0.5)
        # print(distance)
        return distance
    elif len(X)>=len(Y):
        Difference = len(X)-len(Y)
        dist=99999
        for i in range(Difference):
            local=Calcs_Euclidean_distance(X[i:len(Y)],Y)
            if  local<=dist and local>0:
                dist=local
        return dist
    else:
        Difference = len(Y) - len(X)
        # print(Difference)
        dist = 99999
        for i in range(Difference):
            local = Calcs_Euclidean_distance(Y[i:len(X)+i], X)
            if local <= dist and local>0:
                dist = local
        return dist


def Calcs_DTW_distance(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=10000):
    ts_a = np.array(ts_a)
    ts_b = np.array(ts_b)
    ts_a = Z_normal(ts_a)
    ts_b = Z_normal(ts_b)
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    return cost[-1, -1]
if __name__ == '__main__':
    # print(Calcs_Euclidean_distance([1,2],[1,2,3,4,5,6]))
    # print(Calcs_DTW_distance([1,2],[1,2,3,4,5,6]))
    pf = pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\synthetic_data\R15.txt")
    pf1 = pf.iloc[:, 0].tolist()
    data1 = pf1
    pf2 = pf.iloc[:, 1].tolist()
    data2 =pf2[0:300]
    start =time.clock()
    result=Calcs_Euclidean_distance(data1,data2)
    # result=Calcs_DTW_distance(data1,data2)
    print(result)
    elapsed = (time.clock() - start)
    print("测试时间为:{}".format(elapsed))