import time
import numpy as np
import pandas as  pd
from Time_Series.Model_Distance_Calcs import Calcs_Euclidean_distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics

def Clustering_Shapelet_Traditional(D,Slen,k):
    for i in range(len(D)):
        if Slen>len(D[i]):
            return -1
    U_shapelets=[]
    D=pd.DataFrame(D)
    ts=D.iloc[0,:]
    # print(ts)
    while True:
        Gap_and_Dt=[]
        cnt=0
        for i in range(Slen):
            for j in range(len(ts)-i):
                # Calcs_Gap(D, ts[j:j + i + 1],k)
                GAP,DT,TS=Calcs_Gap(D,ts[j:j+i+1],k)
                Gap_and_Dt.append([GAP,DT,TS,j,i])
        # print(Gap_and_Dt)
        max=0
        index=0
        DDt=0
        for i in range(len(Gap_and_Dt)):
            if Gap_and_Dt[i][0]>=max:
                max=Gap_and_Dt[i][0]
                index=i
                DDt=Gap_and_Dt[i][1]
        # print(max,index,DDt)
        U_shapelets.append([Gap_and_Dt[index][2],Gap_and_Dt[index][3],Gap_and_Dt[index][4],Gap_and_Dt[index][1]])
        dis = []
        for i in range(len(D)):
            distance = Calcs_Euclidean_distance(Gap_and_Dt[index][2], D.iloc[i, :])
            dis.append([i, distance])
        # print(dis)
        Da=[]
        Danumber=[]
        for i in range(len(dis)):
            if dis[i][1]<DDt:
                Da.append(dis[i])
                Danumber.append(dis[i][1])
        # print(Da)
        if len(Da) <=1:
            break
        maxdis=0
        maxindex=0
        for i in range(len(dis)):
            if dis[i][1]>maxdis:
                maxdis=dis[i][1]
                maxindex=dis[i][0]
        ts =D.iloc[maxindex,:]
        cita=np.mean(Danumber)+np.var(Danumber)
        P=[]
        for i in range(len(dis)):
            if dis[i][1]>=cita:
                P.append(D.iloc[i,:])
        D=pd.DataFrame(P)
    return U_shapelets


def Calcs_Gap(D,ts,k):
    maxGap=0
    dt=0
    dis=[]

    for i in range(len(D)):
        # print(D.iloc[i,:])
        # print(ts)
        distance=Calcs_Euclidean_distance(ts,D.iloc[i,:])
        dis.append([i,distance])
    # print(dis)
    dis.sort(key=lambda x: x[1])
    # print("*****")
    # print(dis)
    r=0
    for i in range(len(dis)-1):
        DA = []
        DB = []
        if dis[i][1]==0 and dis[i+1][1]==0:
            continue
        d=(dis[i][1]+dis[i+1][1])/2
        for i in range(len(dis)):
            if dis[i][1]<d:
                DA.append(dis[i][1])
            elif dis[i][1]>=d:
                DB.append(dis[i][1])
            # r=(len(DA)+1)/(len(DB)+1)
            if len(DB) == 0:
                continue
            r=len(DA)/len(DB)
            # print(r)
            if (r>(1/k) and r<(1-1/k)) or r>1000:
                gap=np.mean(DB)-np.var(DB)-(np.mean(DA)+np.var(DA))
                if gap>maxGap:
                    maxGap=gap
                    dt=d
        # print(maxGap,"Gap")
        # print(dt,"dt")
        return maxGap,dt,ts

def K_means(D,k):
    cluster = KMeans(n_clusters=k, random_state=0).fit(D)
    return cluster.labels_

def RI(D,P):
    D=np.array(D)
    P=np.array(P)
    result=metrics.adjusted_rand_score(P.flatten(), D.flatten())
    return result

if __name__ == '__main__':
    D = [
        [5, 1, 1, 4, 5, 0],
        [1, 1, 3, 4, 7, 0],
        [1, 4, 5, 8, 9, 1],
        [1, 4, 5, 7, 8, 0],
        [2, 2, 4, 5, 7, 1],
        [1, 8, 9, 3, 4, 1],
        [2, 3, 4, 5, 1, 0],
        [2, 3, 5, 1, 2, 1],
        [2, 3, 4, 5, 0, 1],
        [3, 3, 1, 1, 1, 1],
        [5, 1, 1, 4, 1, 1],
        [1, 3, 4, 5, 1, 7],
        [5, 5, 1, 7, 5, 1],
        [5, 5, 9, 1, 5, 1],
        [5, 1, 7, 1, 5, 0],
        [5, 4, 2, 4, 5, 0],
        [5, 1, 3, 4, 5, 0],
        [5, 3, 1, 3, 5, 1],
        [5, 1, 1, 2, 5, 0],
        [4, 3, 2, 1, 0, 1]
    ]
    DD = [
        [1, 1, 1, 1, 1, 6, 5, 6, 7, 5, 4, 1, 1, 1, 2, 4, 3, 1, 5, 0],
        [1, 1, 2, 2, 1, 6, 5, 8, 7, 5, 4, 1, 1, 1, 2, 5, 4, 2, 4, 0],
        [1, 1, 0, 1, 1, 7, 5, 8, 8, 4, 6, 2, 1, 1, 2, 3, 5, 2, 4, 0],
        [0, 1, 2, 0, 1, 7, 5, 8, 8, 5, 5, 1, 1, 1, 2, 3, 4, 2, 5, 0],
        [7, 7, 7, 3, 4, 1, 1, 1, 0, 2, 5, 4, 5, 6, 3, 2, 9, 9, 1, 1],
        [8, 7, 7, 4, 4, 1, 1, 0, 0, 0, 4, 4, 5, 5, 1, 5, 9, 8, 2, 1],
        [9, 6, 5, 4, 3, 2, 1, 0, 0, 0, 5, 3, 3, 4, 1, 3, 8, 9, 1, 1]
    ]
    pf = pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\cnn_data\cifar_features.txt",header=None)
    print(pf.shape)
    #Aggregation
#测试
    # U_s=[]
    # start = time.clock()
    # result=Clustering_Shapelet_Traditional(DD,10,1000)
    # elapsed = (time.clock() - start)
    # print("测试时间为:{}".format(elapsed))
    # for i in range(len(result)):
    #     print(result[i][0].values)
    #     print(result[i][1])
    #     print(result[i][2])
    #     print(result[i][3])
    #     U_s.append(result[i][0].values)
    # Dt=result[i][3]
    # DA=[]
    # DB=[]
    # for j in range(len(U_s)):
    #     for i in range(len(DD)):
    #         r=Calcs_Euclidean_distance(DD[i],U_s[j])
    #         print()
    #         if r>Dt:
    #             DB.append(DD[i])
    #         else:
    #             DA.append(DD[i])
    # plt.figure()
    # for i in range(len(DA)):
    #     plt.plot(range(len(DD[i])), DA[i], label="DA" + str(i), color="gold")
    # for i in range(len(DB)):
    #     plt.plot(range(len(DD[i])), DB[i], label="DB" + str(i), color="blue")
    # for i in range(len(U_s)):
    #     plt.plot(range(len(U_s[i])),U_s[i],label="U_Shapelet",color='red')
    # plt.xlabel("time_point", fontsize=12)
    # plt.ylabel("data_1D", fontsize=12)
    # plt.title("Shapelet_Clustering", fontsize=20)
    # plt.legend()
    # plt.show()
#测试
    D=pd.DataFrame(pf.iloc[:,0:-1])
    P=pd.DataFrame(pf.iloc[:,-1])
    result=K_means(D,9)
    print(result)
    print(RI(result,P))
    pca = PCA(n_components=2)
    pca = pca.fit(D)
    D_dr = pca.transform(D)
    # A=[]
    # B=[]
    # for i in range(len(D)):
    #     if result[i]==0:
    #         A.append(D.iloc[i,:])
    #     else:
    #         B.append(D.iloc[i,:])
    # A=np.array(A)
    # B = np.array(B)
    # plt.figure()
    # plt.scatter(A[:,0], A[:, 1], color="b")
    # plt.scatter(B[:, 0], B[:, 1], color="r")
    # plt.title("K-means", fontsize=20)
    # plt.show()
#实验
    # A = []
    # B = []
    # C =[]
    # D1= []
    # E=[]
    # F=[]
    # G=[]
    # for i in range(len(result)):
    #     if result[i] == 0:
    #         A.append(D.iloc[i,:])
    #     elif result[i] == 1:
    #         B.append(D.iloc[i,:])
    #     elif result[i]==2:
    #         C.append(D.iloc[i,:])
    #     elif result[i]==3:
    #         D1.append(D.iloc[i,:])
    #     elif result[i]==4:
    #         E.append(D.iloc[i,:])
    #     elif result[i]==5:
    #         F.append(D.iloc[i,:])
    #     else:
    #         G.append(D.iloc[i,:])
    # A,B,C,D1,E,F,G = np.array(A),np.array(B),np.array(C),np.array(D1),np.array(E),np.array(F),np.array(G)
    # print(A)
    # plt.figure()
    # plt.scatter(A[:, 0], A[:, 1])
    # plt.scatter(B[:, 0], B[:, 1])
    # plt.scatter(C[:, 0], C[:, 1])
    # plt.scatter(D1[:, 0], D1[:, 1])
    # plt.scatter(E[:, 0], E[:, 1])
    # plt.scatter(F[:, 0], F[:, 1])
    # plt.scatter(G[:, 0], G[:, 1])
    # plt.title("K-means", fontsize=20)
    # plt.show()
    # A=[]
    # B=[]
    # for i in range(len(D)):
    #     if result[i]==0:
    #         A.append(D.iloc[i,:])
    #     else:
    #         B.append(D.iloc[i,:])
    # A=np.array(A)
    # B = np.array(B)
    # plt.figure()
    # plt.scatter(A[:,0], A[:, 1], color="b")
    # plt.scatter(B[:, 0], B[:, 1], color="r")
    # plt.title("K-means", fontsize=20)
    # plt.show()
    #
