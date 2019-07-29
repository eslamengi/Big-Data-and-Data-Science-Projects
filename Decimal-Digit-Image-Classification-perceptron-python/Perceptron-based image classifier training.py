"""
Created By Eslam Elsheikh

this is to compute the best eta to use to classify each class separately 
at the end we get one wieght vector (10X785) will be used on the test data
to get the best accuracy confusion matrix for the 10 classes

"""

##########################
#Import important packages
##########################

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import misc

#####################
#load training images
#####################

Train_Path='E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Train'
os.chdir(Train_Path)
Training_Labels = np.loadtxt('Training Labels.txt')
files=os.listdir(Train_Path)
files.pop()
files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

data=[]
for i in files:    
    img=misc.imread(i)
    type(img)
    img.shape
    #change dimention to 1 dimentional array instead of (28x28)
    img=img.reshape(784,)
    img=np.append(img,1)
    data.append(img)

#learning rate
eta = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9]

#training labels for each class
    
t0 = [1 if n == 0 else -1 for n in Training_Labels]
t1 = [1 if n == 1 else -1 for n in Training_Labels]
t2 = [1 if n == 2 else -1 for n in Training_Labels]
t3 = [1 if n == 3 else -1 for n in Training_Labels]
t4 = [1 if n == 4 else -1 for n in Training_Labels]
t5 = [1 if n == 5 else -1 for n in Training_Labels]
t6 = [1 if n == 6 else -1 for n in Training_Labels]
t7 = [1 if n == 7 else -1 for n in Training_Labels]
t8 = [1 if n == 8 else -1 for n in Training_Labels]
t9 = [1 if n == 9 else -1 for n in Training_Labels]

t = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]

def perceptron_func(data, t, eta): 
    
    w1= np.zeros(784)
    w = np.append(1, w1)# Weight vector and W0 "Bias"label = [0]
    y = [0]
    misspoint = [0,0,0]
    iteration = 0
    while len(misspoint) != 0:  
        misspoint = []
        label = []
        y = []
        iteration += 1
        #this loop to calculate Y(X) for wach data point
        for i in range(len(data)):
            y = np.append(y, np.dot(w, data[i]))
        
        #print('iteration # ' + str(iteration) + '  y(x) =     ' + str(y))
        
        '''this loop to calculate new label based on Y(X), 
        if >=0 then label = 1 if <0 then label = -1'''
        
        for n in range(len(y)):
            if y[n] >= 0:
                label.append(1)
            else:
                label.append(-1)
            
        #print('iteration # ' + str(iteration) + '  calc. lable =     ' + str(label))
        
        '''compare new label with original data label to find miss classified point
        if label != t then this point is miss classified'''
        
        for m in range(len(label)):
            if t[m] != label[m]:
                misspoint = data[m]
                Error = misspoint * t[m]
                #print('iteration # ' + str(iteration) + '  missclassified point =   ' + str(misspoint))
                #print('=============================================================================')
            
                w = w + (eta * Error)
                
                break
            
        
   #print('=============================================================================')
    print('number of iterations =   ' + str(iteration))   
    
    return w
    
sequence = 0 #index to save wieght output txt file
w_eta = []

#this loop to run the perceptron function to interate over all eta values to train the classifier

for l in range(len(eta)):
    print('Training algorith for eta =      ' + str(eta[l]) + '\n')
    for i in range(len(t)):
        Wieght = perceptron_func(data, t[i], eta[l])
        w_eta.append(Wieght)
    wieghtname = ('w_eta_test_' + str(sequence) + '.txt')
    sequence += 1
    Weightpath = 'E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Training Wieght Vectors'
    os.chdir(Weightpath)
    np.savetxt(wieghtname, w_eta)  