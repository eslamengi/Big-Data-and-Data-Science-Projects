'''
Created By Eslam Elsheikh

this is to compute the Confusion matrix for the test data
and also compute the best accuracy confusion matrix after applying the wieght vectors to
valaditaion data
'''

##########################
#Import important packages
##########################

import numpy as np
import os                   
from scipy import misc
import scipy.misc
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import itertools


#################
#load test images
#################
TestPath='E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Test'
os.chdir(TestPath)
Test_Labels = np.loadtxt('Test Labels.txt')
filestest=os.listdir(TestPath)
filestest.pop()
filestest = sorted(filestest,key=lambda x: int(os.path.splitext(x)[0]))

all_test=[]
for i in filestest:    
    imgtest=misc.imread(i)
    type(imgtest)
    imgtest.shape
    #change dimention to 1 dimentional array instead of (28x28)
    imgtest=imgtest.reshape(784,)
    imgtest=np.append(imgtest,1)
    all_test.append(imgtest)


############################
#Load Wieght Vector per eta
############################
    
Weightpath = 'E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Training Wieght Vectors'
os.chdir(Weightpath)

#Wx is wieght matrix (10X785) computed for different eta
W0 = np.loadtxt('w_eta_0.txt')
W1 = np.loadtxt('w_eta_1.txt')
W2 = np.loadtxt('w_eta_2.txt')
W3 = np.loadtxt('w_eta_3.txt')
W4 = np.loadtxt('w_eta_4.txt')
W5 = np.loadtxt('w_eta_5.txt')
W6 = np.loadtxt('w_eta_6.txt')
W7 = np.loadtxt('w_eta_7.txt')
W8 = np.loadtxt('w_eta_8.txt')
W9 = np.loadtxt('w_eta_9.txt')

seq = ['-0','-1','-2','-3','-4','-5','-6','-7','-8','-9']#used for naming the confusion matrix

############################
#computing y(x) for each eta
############################

def perceptron_test(all_test, w, Test_Labels, seq):
    
    '''
    This function calculates the classified label for test data pattern
    it computes y(x) for all valyes of W then return a new label matrix 
    each label is MAX(index(y(x)))
    finally; it computes the confusion matrix
    
    it takes 3 arguments as input
    
    1- all_test : represent the input test data pattern data
    2- w : represent the weight vector "decision boundary parameters"
    3- Test_labels : represent the test data label 
    4- seq : define the confusion matrix auto naming sequence index
    
    '''
    
    y=[]
    for i in range(len(w)):
        for j in range(len(all_test)):
            y_init=np.dot(w[i], all_test[j])
            y.append(y_init)
    
    y = np.reshape(y,[200,10], order = 'F')
    predict_label = []
    for n in range(len(y)):
        index = np.argmax(y[n])
        predict_label.append(index)
    
    matrix=confusion_matrix(predict_label, Test_Labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Acc = np.trace(matrix)/2
    
    
    ###########################################
    #Plotting Confusion Matrix in better format
    ###########################################
    
    plt.imshow(matrix,cmap=plt.cm.YlGn,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix of eta 10^ ' + str(seq))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(Test_Labels))) # length of classes
    class_labels = ['0','1','2','3','4','5','6','7','8','9']
    tick_marks
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    
    # plotting text value inside cells
    
    thresh = matrix.max() / 2
    for i,j in itertools.product(range(matrix.shape[0]),range(matrix.shape[1])):
        plt.text(j,i,format(matrix[i,j],'d'),horizontalalignment='center',color='white' if matrix[i,j] >thresh else 'black')
    
    # plotting the Matrix output and saving it to JPG format
    Confusionpath = 'E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Test Confusion matrices'
    os.chdir(Confusionpath)
    confusion_name = ('Confusion' + str(seq) + '.jpg')
    plt.savefig(confusion_name, facecolor = None, transparent = True)
    plt.show(); 
    print('Accuracy at eta 10^' + str(seq) + ' =   ' + str(Acc) + ' % ')

    return matrix


#####################################
#computing confusion_eta_index matrix
#####################################
    
W = [W0, W1, W2, W3, W4, W5, W6, W7, W8, W9]

for m in range(len(W)):
    
    Confusion = perceptron_test(all_test, W[m], Test_Labels, seq[m])
    
#########################################
#computing best accuracy confusion matrix
#########################################

Weightpath = 'E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Training Wieght Vectors'
os.chdir(Weightpath)
eta_best = np.loadtxt('w_eta_best.txt')
eta_best.reshape(10,1)

W_best = []
for i in range(len(W)):
    # collecting best eta wieght for each class to compute the best accuracy matrix
    wieght = W[int(eta_best[i])][i] 
    W_best.append(wieght)

print(np.shape(W_best))
    
Confusion = perceptron_test(all_test, W_best, Test_Labels, '-b')