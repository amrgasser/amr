import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from matplotlib import pyplot

data = pd.read_csv(r"/Users/amr_g/Desktop/Machine Learning/Assignment 3/house_data_complete.csv").dropna().to_numpy()

train,validate,test = np.split(data,[int(.6*len(data)),int(.8*len(data))])
validate1,test1,train1 = np.split(data,[int(.2*len(data)),int(.4*len(data))])
test2,validate2,train2 = np.split(data,[int(.2*len(data)),int(.4*len(data))])
price = train[:,2]
train= train[:,[3,6,10,11]]

tprice = test[:,2]
test= test[:,[3,6,10,11]]

price1 = train1[:,2]
train1= train1[:,[3,6,10,11]]

tprice1 = test1[:,2]
test1= test1[:,[3,6,10,11]]

price2 = train2[:,2]
train2= train2[:,[3,6,10,11]]

tprice2 = test2[:,2]
test2= test2[:,[3,6,10,11]]


bedrooms = train[:,0]
size = train[:,1]
condition = train[:,2]
grade = train[:,3]
names = ['bedrooms' ,'size','condition', 'grade','error']

def plotData(bedrooms, price,i):
    fig = pyplot.figure()
    pyplot.plot(bedrooms, price, 'ro', ms=10, mec='k')
    pyplot.ylabel('price')
    pyplot.xlabel(names[i])
    pyplot.show()


def  featureNormalize(X):
    X_norm = X.copy()
    mean = np.zeros(X.shape[0])
    mean = np.mean(X , axis=0)

    X_norm = X/mean

    return X_norm, mean


train_Xnorm, train_mu = featureNormalize(train)
test_Xnorm, test_mu = featureNormalize(test)


ons = np.ones((price.size,1))
ons2 = np.ones((tprice.size,1))

train_Xnorm = np.concatenate([ons,train_Xnorm,],axis=1)
test_Xnorm = np.concatenate([ons2,test_Xnorm,],axis=1)

train_Xnorm1, train_mu1 = featureNormalize(train1)
test_Xnorm1, test_mu1 = featureNormalize(test1)


ons = np.ones((price1.size,1))
ons2 = np.ones((tprice1.size,1))

train_Xnorm1 = np.concatenate([ons,train_Xnorm1,],axis=1)
test_Xnorm1= np.concatenate([ons2,test_Xnorm1,],axis=1)


train_Xnorm2, train_mu2 = featureNormalize(train2)
test_Xnorm2, test_mu2 = featureNormalize(test2)


ons = np.ones((price2.size,1))
ons2 = np.ones((tprice2.size,1))

train_Xnorm2 = np.concatenate([ons,train_Xnorm2,],axis=1)
test_Xnorm2 = np.concatenate([ons2,test_Xnorm2,],axis=1)





def computeCost(X, y, theta):
    m = y.size
    theta = np.array(theta)

    J = 0
    H = np.dot(X,theta.T)

    Diff = H - y
    J = (1/(2*m))*(np.sum(np.square(Diff)))

    return J
theta = np.zeros(5)
a = 0.01
iterations = 10000
def iteration(it):
    ite = np.zeros(iterations)
    for i in range(it):
        ite[i] = i

    return ite

j=computeCost(train_Xnorm,price,theta)

jtest = computeCost(test_Xnorm,tprice,theta)
print(jtest)

def gradientDescent(X, y, theta, alpha, num_iters ):

    m = y.shape[0]
    theta = theta.copy()

    J_history = []


    for i in range(num_iters):
        sumofh0x = np.dot(X, theta)
        theta = theta - ((alpha / m) * (np.dot(X.T, sumofh0x - y)))
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

theta, Jhist = gradientDescent(train_Xnorm,price,theta,a,iterations)
theta1, Jhist1 = gradientDescent(train_Xnorm1,price1,theta,a,iterations)
theta2, Jhist2 = gradientDescent(train_Xnorm2,price2,theta,a,iterations)
avg = (Jhist[9999] +Jhist2[9999]+Jhist1[9999])/3
print(avg)
x = iteration(iterations)
plotData(Jhist,x,4)