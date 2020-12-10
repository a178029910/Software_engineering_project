import numpy
import pandas
import matplotlib.pyplot as plt
from Time_Series.Model_Distance_Calcs import *
from Time_Series.Z_normal import *
from Time_Series.Forecast import *
from Time_Series.Sampling import *
from Time_Series.Clustering import *
from Time_Series.Classfication import *

# class Data:
#     def __init__(self,D):
#         self.D=D

if __name__ == '__main__':
    pf = pd.read_csv(r"C:\Users\HASEE\Desktop\BorderPeelingClustering-master\synthetic_data\R15.txt")
    pf1 = pf.iloc[:, 0].tolist()
    data1 = pf1
    print(Simple_Sampling(data1, 5))
    print(PAA_Sampling(data1, 5))
    print(L_Sampling(data1, 2))