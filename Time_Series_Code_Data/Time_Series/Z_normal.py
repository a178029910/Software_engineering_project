'''
Z-normal
'''
def Z_normal(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total)/lenth
    tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
    tempsum = pow(float(tempsum)/lenth,0.5)
    if tempsum==0:
        return data
    for i in range(lenth):
        data[i] = (data[i] - ave)/tempsum
    return data


if __name__ == '__main__':
    data =[2,2,3,4,5,6,7,8,11,21,32,55,11]
    print(Z_normal(data))