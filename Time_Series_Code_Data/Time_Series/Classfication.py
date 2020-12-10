import time

import numpy as np
import pandas as pd
from sklearn import tree
from Time_Series.Model_Distance_Calcs import Calcs_Euclidean_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def acu_curve(y, prob):
    # y真实prob预测
    print(type(y))
    for i in range(len(y)):
        if y.values[i] !=0 and y.values[i]!=1:
            y.values[i]= 0
        if prob[i] !=0 and prob[i] !=1:
            prob[i] = 0
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8),dpi=100)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")

    plt.show()
def Decision_tree_Train_Predict(D,P):
    D=pd.DataFrame(D)
    P=pd.Series(P)
    Target = D.iloc[:,5]
    Data = D.iloc[:,0:-1]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Data, Target, test_size=0.1)
    clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=3)
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest,Ytest)
    list = clf.predict(P.values.reshape(1,-1))
    return score,list

def Random_Forest_Train_Predict(D,P):
    D = pd.DataFrame(D)
    P = pd.DataFrame(P)
    Target = D.iloc[:, -1]
    Data = D.iloc[:, 0:-1]
    rfc = RandomForestClassifier(n_estimators=25, oob_score=True)
    rfc = rfc.fit(Data, Target)
    result=rfc.predict(P)
    return result

def Calcs_max_gain(list,D,P,maxGain,maxGap,ED):
    index_list=[]
    for i in range(len(list)):
        index_list.append([i,list[i],P[i]])
    index_list.sort(key=lambda x:x[1])
    dt=0
    best_dt=-1
    update=False
    # print(index_list)
    for i in range(len(D)-1):
        # print(index_list[i][1])
        if index_list[i][1]==0 and index_list[i+1][1]==0:
            continue
        dt=(index_list[i][1]+index_list[i+1][1])/2
        DA=[]
        DAg=0
        DB=[]
        DBg=0
        for i in range(len(index_list)):
            if index_list[i][1]<=dt:
                DA.append(index_list[i][2])
                DAg+=index_list[i][1]
            else:
                DB.append(index_list[i][2])
                DBg+=index_list[i][1]
        # print(DA,"*****da",DAg)
        # print(DB,"*******db",DBg)
        # print("********")
        # print((-1 / len(D) * Calcs_main(DA)))
        # print('********GaDA')
        # print(-1 / len(D) * Calcs_main(DB))
        # print('********GaDB')
        Gain=ED-1/len(D)*Calcs_main(DA)-1/len(D)*Calcs_main(DB)
        # print(Gain)
        if len(DB)!=0 and len(DA)!=0:
            Gap = (1 / len(DB) * DBg) - (1 / len(DA) * DAg)
        else:
            Gap=-1
        if Gain>maxGain or (Gain==maxGap and Gap>maxGap):
            maxGain=Gain
            maxGap=Gap
            update=True
            best_dt = dt
        else:
            update=False
    return best_dt,update,maxGap,maxGain

def Classfication_Shapelet_Train_Predict_Traditional(D,C):
    D=np.array(D)
    P=np.array(D[:,-1])
    # print(P)
    D=np.array(D[:,0:-1])
    # print(D)
    ED=Calcs_ED(P)
    # print(ED)
    for i in range(len(D)):
        if C>len(D[i]):
            return -1
    maxGain=0
    maxGap=0
    best_s=[]
    dt=0
    i1,k1,j1=-1,-1,-1
    for i in range(len(D)):
        # print(len(D))
        for j in range(C):
            for k in range(len(D[i])-j):
                list = []
                for l in range(len(D)):
                    sdist=Calcs_Euclidean_distance(D[i][k:k+j+1],D[l])
                    list.append(sdist)
                # print(list)
                best_dt,update,Gap,Gain=Calcs_max_gain(list,D,P,maxGain,maxGap,ED)
                if update:
                    maxGap=Gap
                    maxGain=Gain
                    dt=best_dt
                    best_s=D[i][k:k+j+1]
                    i1,k1,j1=i+1,k+1,j+1
    # print(best_s)
    print("这条时间序列为第{}条,从第{}点开始,长度为{},注意啦!".format(i1,k1,j1))
    return best_s,i1,k1,j1,dt

def Calcs_ED(P):
    P=pd.Series(P)
    lenth=len(P)
    count=P.value_counts()
    Gain=0
    for i in range(len(count)):
        Gain+=(count.iloc[i]/lenth)*np.log2(count.iloc[i]/lenth)
    return -Gain

def Calcs_main(P):
    P = pd.Series(P)
    length=len(P)
    count=P.value_counts()
    # print(type(count))
    # print("**********")
    # print(length)
    # print("**********")
    # print(count.values)
    # print("**********")
    # print(count)
    # print("**********")
    # print(count.iloc[0])
    # print("**********")
    Gain=0
    if len(count)==0:
        return 0
    elif len(count)==1:
        # Gain+=
        return Gain
    else:
        for i in range(len(count)):
            zhi=((count.iloc[i]/length)*np.log2(count.iloc[i]/length))
            # print(zhi)
            Gain=Gain+zhi
    # print(Gain)
    return -Gain*length

def information(zhen,yuce):
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(len(zhen)):
        if zhen.values[i]==yuce[i]:
            if zhen.values[i] !=1:
                TN+=1
            else:
                TP+=1
        else:
            if zhen.values[i] !=1:
                FP+=1
            else:
                FN+=1
    print(TP)
    print(FP)
    print(FN)
    print(FP)

    P=TP/(TP+FP)
    R = TP / (TP + FN)
    return P,R

if __name__ == '__main__':
    D=[
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
    DD=[
        [1, 1, 1, 1, 1, 6, 5, 6, 7, 5, 4, 1, 1, 1, 2, 4, 3, 1, 5, 0],
        [1, 1, 2, 2, 1, 6, 5, 8, 7, 5, 4, 1, 1, 1, 2, 5, 4, 2, 4, 0],
        [1, 1, 0, 1, 1, 7, 5, 8, 8, 4, 6, 2, 1, 1, 2, 3, 5, 2, 4, 0],
        [0, 1, 2, 0, 1, 7, 5, 8, 8, 5, 5, 1, 1, 1, 2, 3, 4, 2, 5, 0],
        [7, 7, 7, 3, 4, 1, 1, 1, 0, 2, 5, 4, 5, 6, 3, 2, 9, 9, 1, 1],
        [8, 7, 7, 4, 4, 1, 1, 0, 0, 0, 4, 4, 5, 5, 1, 5, 9, 8, 2, 1],
        [9, 6, 5, 4, 3, 2, 1, 0, 0, 0, 5, 3, 3, 4, 1, 3, 8, 9, 1, 1]
    ]
    P=[3,3,4,5,1]
    # Score,list=Decision_tree_Train_Predict(D,P)
    # print(Random_Forest_Train_Predict(D,P))
    # D2=[
    #     [1,1,1,1],
    #     [2,2,2,2]
    # ]
    # D=np.array(D)
    # C=np.random.randint(1,20,size=(25,25))
    # print(C)
    # print(Classfication_Shapelet_Train_Predict_Traditional(D,5))
    # print(Calcs_main([1,1,0]),"123")
    # print(Calcs_ED([1,1]))
    # a = [['USA', 'b'], ['China', 'c'], ['Canada', 'd'], ['Russia', 'a']]
    # a.sort(key=lambda x: x[1], reverse=False)
    # print(a)
#测试随机森林
    # pf = pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\synthetic_data\flame.txt")
    # data2=pf.iloc[:,0:2]
    # target2=pf.iloc[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(data2, target2, test_size=0.3, random_state=0)
    # X_train=X_train.join(y_train)
    # y=Random_Forest_Train_Predict(X_train, X_test)
    # print(y)
    # acu_curve(y_test, y)
    # print(information(y_test,y))
#测试shapelet
    # D=np.array(D)
    # pf = pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\synthetic_data\flame.txt")
    # # C=np.random.randint(1,20,size=(15,15))
    # print(Classfication_Shapelet_Train_Predict_Traditional(D,5))
    # print(Classfication_Shapelet_Train_Predict_Traditional(DD,5))
    # plt.figure(figsize=(10,8),dpi=100)
    # color=["red","gold","green","cyan","slategray","plum","b"]
    # for i in range(len(DD)):
    #     plt.plot(range(len(DD[i])),DD[i],label="DD"+str(i),color=color[i])
    # plt.xlabel("time_point",fontsize=12)
    # plt.ylabel("data_1D",fontsize=12)
    # plt.title("D3",fontsize=20)
    # plt.legend()
    # plt.show()
    # start = time.clock()
    # print(Classfication_Shapelet_Train_Predict_Traditional(DD, 10))
    # elapsed = (time.clock() - start)
    # print("测试时间为:{}".format(elapsed))
    dt=0.6697811566101117
    DD=pd.DataFrame(DD)
    DDD=pd.DataFrame(DD.iloc[:,0:-1])
    DDP=pd.DataFrame(DD.iloc[:,-1])
    DA=[]
    DB=[]
    for i in range(len(DDD)):
        dis=Calcs_Euclidean_distance(DDD[i],[1, 1, 1, 1, 6, 5, 6, 7, 5, 4])
        if dis>dt:
            DA.append(DD.iloc[i,:])
        else:
            DB.append(DD.iloc[i,:])
    plt.figure(figsize=(10, 8), dpi=100)
    # color=["red","gold","green","cyan","slategray","plum","b"]
    for i in range(len(DA)):
        plt.plot(range(len(DA[i])),DA[i],label="DA"+str(i),color="gold")
    for i in range(len(DB)):
        plt.plot(range(len(DB[i])),DB[i],label="DB"+str(i),color="blue")
    plt.plot(range(1,11),[1, 1, 1, 1, 6, 5, 6, 7, 5, 4],label="Shapelet",color='red')
    plt.xlabel("time_point",fontsize=12)
    plt.ylabel("data_1D",fontsize=12)
    plt.title("Shapelet_Classification",fontsize=20)
    plt.legend()
    plt.show()
