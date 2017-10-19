#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:11:00 2017

@author: Chris

Hedging Assignment
"""

#Import a set of potentially useful packages
import pandas as pd
import numpy
import pylab as P
import math
import random
from sklearn import linear_model
from scipy.optimize import curve_fit
import itertools

'''
Cleans the dataset data by linear extrapolating 
from len_ext before the nan/missing
value
'''
def clean(data):
    for j in range(2,len(data)):
        #Is this entry empty?
        if numpy.isnan(data[j]):
            data[j] = data[j-1] + (data[j-1] - data[j-2])
        else:
            pass
    return data

'''
Uses num_points to extrapolate the data forward
by time T. 
Returns the predicted differential.
Note: index must be greater than num_points
'''
def predict(data_x, data_y, T, num_points, index):
    #Now want to use some data to predict the behaviour after time T
    #Train on previous num_points data points
    #start at index
    #prev = num_points
    x_slice = data_x[index-num_points:index]
    y_slice = data_y[index-num_points:index]
    #Fit data linear
    reg = linear_model.LinearRegression()
    reg.fit(x_slice,y_slice)
    return reg.predict(data_x)[index+T] - data_y[index]


'''
Simple moving average from a slice of data
Calculates the average of the past num_points working
backwards from index
'''
def sma(y_slice):
    tot = 0.
    for i in range(len(y_slice)):
        tot += y_slice[i]
    return tot/len(y_slice)
    
'''
Exponential moving average from a slice of data
Calculates weights 2/(num_points+1)
EMA = weight*true_val+(1-weight)*prev_EMA Note: first EMA = true_val
'''
def ema(y_slice):
    weight = 2./(len(y_slice)+1)
    tot = y_slice[0]*((1.-weight)**len(y_slice))
    for i in range(len(y_slice)):
        tot += weight*(y_slice[len(y_slice)-1-i]*(1-weight)**i)
    return tot

'''
Predict the value of the security using the sma
See below for more details   
'''
def predict_sma(data_y, T, num_points, index):
    y_slice = data_y[index-num_points:index]
    for i in range(T):
        y_slice = numpy.append(y_slice,sma(y_slice))
        y_slice = numpy.delete(y_slice,0)
    return y_slice[len(y_slice)-1] - data_y[index] 

'''
Predict the value of the security using the ema
Returns the predicted value on the Tth day after the index
Assume the first ema predicts the price for one later day
and then use that prediction in the next iteration 
until T days are complete
Return the predicted difference as before
'''
def predict_ema(data_y, T, num_points, index):
    y_slice = data_y[index-num_points:index]
    for i in range(T):
        y_slice = numpy.append(y_slice,ema(y_slice))
        y_slice = numpy.delete(y_slice,0)
    return y_slice[len(y_slice)-1] - data_y[index]

'''
Returns the true difference in the data set
'''
def true_diff(data_y, T, index):
    return data_y[index+T] - data_y[index]

'''
Finds the basket of least volatile stocks that match the change delta
Takes x_data, y_data, time for extrapolation, num_points of history
index at time of prediction, delta which is the change in target
and num_cut which is the number of least volatile stocks to consider (reduce to speed up)
'''
def find_basket(x_data,data_table,T,num_points,index,delta,num_cut,tolerance):
    #Success flag
    found_flag = 0
    #First calculate how volatile each stock appears
    volatility = numpy.zeros(len(data_table))
    for i in range(len(data_table)):
        volatility[i] = numpy.std(data_table[i][index-num_points:index])
    vol_ind = numpy.argsort(volatility)
    #Ok now find four securities to add to match target
    store = numpy.zeros(num_cut)
    for i in range(num_cut):
        store[i] = predict_ema(data_table[vol_ind[i]].reshape(-1,1),T,num_points,index)
    
    best_com = list()
    #Look at all permutations of 4 securities and find sum closest to delta until found
    count = 0
    while found_flag != 1:
        for p in itertools.combinations(store,4):
            if math.fabs(sum(p)-delta)<(tolerance+count*tolerance):
                best_com = p
                found_flag = 1
                break
        count += 1
    #What are the indices of the best combination
    delta_ind = []
    for i in range(len(store)):
        for num in best_com:
            if num == store[i]:
                delta_ind = numpy.append(delta_ind,i)
    
    #Match indicices 
    basket_ind = numpy.zeros(4)
    for i in range(len(delta_ind)):
        basket_ind[i] = vol_ind[int(delta_ind[i])]
    return basket_ind

#############################################################3
'''
Playing with code starting here
In production would include this in main()
'''


'''
Load and clean data
'''
#Skip header
universe = pd.read_csv('./data/X.csv', header=0)
target = pd.read_csv('./data/y.csv', header=0)
category = pd.read_csv('./data/categorical.csv', header=0)
num_uni = len(universe.values[0,:])-2
sec_data_table = numpy.zeros((num_uni,262))
tar_data_table = numpy.zeros((4,262))
for i in range(1,num_uni+1):
    sec_data_table[i-1] = clean(universe.iloc[:,i].values)
for i in range(1,5):
    tar_data_table[i-1] = clean(target.iloc[:,i].values)
        
x = numpy.arange(len(universe['Date'])).reshape(-1,1)


##########################################

'''
Look at residuals using linear extrapolation
'''
x_array = x
prev_points = 100 #preceding points
Ttest = 5 #prediction time
data_index1 = 1
data_index2 = 6
residual1 = numpy.zeros(len(x)-prev_points-Ttest)
residual2 = numpy.zeros(len(x)-prev_points-Ttest)
xvals = numpy.zeros(len(x)-prev_points-Ttest)
for i in range(prev_points,len(x)-Ttest):
    xvals[i-prev_points] = i
    residual1[i-prev_points] = true_diff(sec_data_table[data_index1],Ttest,i)-predict(x_array,sec_data_table[data_index1],Ttest,prev_points,i)
    residual2[i-prev_points] = true_diff(sec_data_table[data_index2],Ttest,i)-predict(x_array,sec_data_table[data_index2],Ttest,prev_points,i)

#Fit residuals using a sinusoid
def sinusoid(x,*p):
    return p[0]*numpy.sin(p[1]*x+p[2])

popt, pcov = curve_fit(sinusoid,xvals,residual2,p0=(1,0.06,0.))
print 1./(2.*math.pi*popt[2])

#####################################
       
'''
Look at how the exponential moving average and simple moving average behave
'''
x_array = x
prev_points = 10 #preceding points
Ttest = 5 #prediction time
data_index1 = 1
data_index2 = 6
sma1 = numpy.zeros(len(x)-prev_points-Ttest)
ema1 = numpy.zeros(len(x)-prev_points-Ttest)
xvals = numpy.zeros(len(x)-prev_points-Ttest)
for i in range(prev_points,len(x)-Ttest):
    xvals[i-prev_points] = i
    y_slice = sec_data_table[data_index2][i-prev_points:i]
    sma1[i-prev_points] = sma(y_slice)
    ema1[i-prev_points] = ema(y_slice)

P.plot(xvals,sma1,'-b')
P.plot(xvals,ema1,'-r')
P.plot(xvals,sec_data_table[data_index2][prev_points:len(x)-Ttest],'.b')
P.show()

#####################

'''
Compare residuals using different predictive models
'''
x_array = x
prev_points = 25 #preceding points
Ttest = 5 #prediction time
data_index1 = 1
data_index2 = 6
residual1 = numpy.zeros(len(x)-prev_points-Ttest)
residual2 = numpy.zeros(len(x)-prev_points-Ttest)
residual3 = numpy.zeros(len(x)-prev_points-Ttest)
xvals = numpy.zeros(len(x)-prev_points-Ttest)
for i in range(prev_points,len(x)-Ttest):
    xvals[i-prev_points] = i
    residual1[i-prev_points] = true_diff(sec_data_table[data_index1],Ttest,i)-predict_ema(sec_data_table[data_index1],Ttest,prev_points,i)
    residual2[i-prev_points] = true_diff(sec_data_table[data_index1],Ttest,i)-predict(x_array,sec_data_table[data_index1],Ttest,prev_points,i)
    residual3[i-prev_points] = true_diff(sec_data_table[data_index1],Ttest,i)-predict_sma(sec_data_table[data_index1],Ttest,prev_points,i)

#Plotting
P.plot(xvals,residual1,'.r')
P.plot(xvals,residual2,'.b')
P.plot(xvals,residual3,'.g')
P.ylim([-5,5])
P.xlabel('Trading Day Since Open')
P.ylabel('Residual')
P.show()

###################################

#Validate finding the basket
num_iter = 100 #Number of trials to run
prev_points = 25 #preceding data to consider
T = 2 #prediction time
num_cut = 20 #number of least volatile securities
tolerance = 0.001 #tolerance in delta
out = numpy.zeros((num_iter,3))
for i in range(num_iter):
    index = random.randint(prev_points,len(x)-3.*T) #random trading day to start
    basket_indices = numpy.random.choice(len(sec_data_table),4) #random basket to start
    gross = 0.
    #How much does the current basket cost
    basket1 = 0.
    for j in basket_indices:
        basket1 += sec_data_table[int(j)][int(index)]
    #How much does the current target cost
    tar1 = 0.
    for j in range(4):
        tar1 += tar_data_table[int(j)][int(index)]
    #Buy taget and sell basket
    gross += (+basket1 - tar1)
    #Predicted change in target
    delta = 0.
    for j in range(4):
        #predict change in target
        delta += predict_ema(tar_data_table[int(j)],T,prev_points,index)
    #Wait for T days
    index += T
    #Choose new basket to track target
    basket_indices = find_basket(x,sec_data_table,T,prev_points,index,delta,num_cut,tolerance)
    #Cost of best basket
    basket2 = 0.
    for j in basket_indices:
        basket2 += sec_data_table[int(j)][int(index)]
    #Cost of target
    tar2 = 0.
    for j in range(4):
        tar2 += tar_data_table[int(j)][int(index)]
    #Cost of that action
    gross += (-basket2 + tar2)
    index += T
    #How much does the current basket cost
    basket3 = 0.
    for j in basket_indices:
        basket3 += sec_data_table[int(j)][int(index)]
    #How much does the current target cost
    tar3 = 0.
    for j in range(4):
        tar3 += tar_data_table[int(j)][int(index)]
    #Buy taget and sell basket
    gross += (+basket3 - tar3)
    #True change in target, Tracked change in basket, predicted change in target
    out[i] = [tar2-tar1, basket3-basket2, delta]
    
#Plotting
P.plot(numpy.arange(len(out)),out[:,0]-out[:,1],'.')
P.xlabel('Model Iteration')
P.ylabel('True target - Basket')
P.ylim([-2,2])
P.show()
P.plot(numpy.arange(len(out)),out[:,1]-out[:,2],'.')
P.xlabel('Model Iteration')
P.ylabel('Pred. target - Basket')
P.ylim([-2,2])
P.show()
print sum(numpy.abs(out[:,0]-out[:,1]))/num_iter


##################################
#Try the algorithm now picking baskets
#Hold gross
basket_indices = numpy.random.choice(len(sec_data_table),4) #random basket to start
gross = 10 #starting investment
i = 25 #starting day
T = 5 #prediction time
tolerance = 0.001 #tolerance in basket
num_points = 25 #preceding data
num_cut = 20 #number of least volatile securities (reduce for speed)
delta = 0 #starting difference in target
while i<len(x)-2*T:
    #Start by buying the target and selling the basket
    #Cost of basket
    basket = 0.
    for j in basket_indices:
        basket += sec_data_table[int(j)][int(i)]
    #Cost of target
    tar = 0.
    for j in range(4):
        tar += tar_data_table[int(j)][int(i)]
    #print basket, tar
    #Buy taget and sell basket
    gross += (+basket - tar)
    #What is the predicted change in the target security
    delta = 0.
    for j in range(4):
        #predict change in target
        delta += predict_ema(tar_data_table[int(j)],T,num_points,i)
    #Wait for five days
    i += T
    #Sell target and buy basket
    #Choose new basket to maximize profit
    basket_indices = find_basket(x,sec_data_table,T,num_points,i,delta,num_cut,tolerance)
    #Cost of basket
    basket = 0.
    for j in basket_indices:
        basket += sec_data_table[int(j)][int(i)]
    #Cost of target
    tar = 0.
    for j in range(4):
        tar += tar_data_table[int(j)][int(i)]
    #print basket, tar
    #Cost of that action
    gross += (-basket + tar)
    i += T
    
print gross