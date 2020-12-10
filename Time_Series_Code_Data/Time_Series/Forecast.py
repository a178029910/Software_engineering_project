from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# index = pd.date_range(start='20201206',periods=50)
# df= pd.Series(np.random.randint(0,15,size=50),index=index)
# pmax = int(len(df) / 10)  # 一般阶数不超过 length /10
# qmax = int(len(df) / 10)
# bic_matrix = []
# for p in range(pmax + 1):
#     temp = []
#     for q in range(qmax + 1):
#         try:
#             temp.append(ARIMA(df, (p, 1, q)).fit().bic)
#         except:
#             temp.append(None)
#         bic_matrix.append(temp)
# bic_matrix = pd.DataFrame(bic_matrix)  # 将其转换成Dataframe 数据结构
# p, q = bic_matrix.stack().idxmin()  # 先使用stack 展平， 然后使用 idxmin 找出最小值的位置
# print(u'BIC 最小的p值 和 q 值：%s,%s' % (p, q))  # BIC 最小的p值 和 q 值：0,1
# 所以可以建立ARIMA 模型，ARIMA(0,1,1)
# model = ARIMA(df, (1, 1, 1)).fit()
def ARIMA_Forecast(T,k):
    model = ARIMA(T, (1, 1, 1)).fit()
    result=model.forecast(k)[0]
    return result
# model.summary2()  # 生成一份模型报告
# print(model.forecast(20)[0])  # 为未来5天进行预测， 返回预测结果， 标准误差， 和置信区间
if __name__ == '__main__':
    pf = pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\synthetic_data\R15.txt")
    pf1 = pf.iloc[:, 0].tolist()
    data1 = pf1
    pf2 = pf.iloc[:, 1].tolist()
    data2 = pf2
    result1 = ARIMA_Forecast(data1,50)
    result2 = ARIMA_Forecast(data2,50)
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(range(len(pf1)), pf1, color="b", label="Origin_data1")
    plt.plot(range(len(pf2)), pf2, color="c", label="Origin_data2")
    # plt.plot(range(len(r1)),r1,color='g',label="Simple_Sampling")
    # plt.plot(range(len(r1),len(r2+r1)), r2, color='r',label="PAA_Sampling")
    plt.plot(range(len(pf1),len(pf1)+50), result1, color='r', label="ARIMA_model1")
    plt.plot(range(len(pf2), len(pf2) + 50), result2, color='g', label="ARIMA_model2")
    plt.xlabel("time_point", fontsize=12)
    plt.ylabel("data_1D", fontsize=12)
    plt.title("Forecast", fontsize=20)
    plt.legend()
    plt.show()
