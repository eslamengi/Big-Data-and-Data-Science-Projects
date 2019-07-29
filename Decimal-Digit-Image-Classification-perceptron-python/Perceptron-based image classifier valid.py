"""
Created By Eslam Elsheikh

this is to compute the best eta to use to classify each class separately 
at the end we get one wieght vector (10X785) will be used on the test data
to get the best accuracy confusion matrix for the 10 classes

"""
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
#load validation images
#################
ValidationPath='E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Validation'
os.chdir(ValidationPath)
Validation_Lables = np.loadtxt('Validation Labels.txt')
filesvalid=os.listdir(ValidationPath)
filesvalid.pop()
filesvalid = sorted(filesvalid,key=lambda x: int(os.path.splitext(x)[0]))

all_validation=[]
for i in filesvalid:    
    imgvalidation=misc.imread(i)
    type(imgvalidation)
    imgvalidation.shape
    #change dimention to 1 dimentional array instead of (28x28)
    imgvalidation=imgvalidation.reshape(784,)
    imgvalidation=np.append(imgvalidation,1)
    all_validation.append(imgvalidation)


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

def perceptron_validation(all_validation, w, Validation_Lables, seq):
    '''
    This function calculates the classified label for validation data pattern
    it computes y(x) for all valyes of W then return a new label matrix 
    each label is MAX(index(y(x)))
    finally; it computes the confusion matrix
    
    it takes 3 arguments as input
    
    1- all_validation : represent the input validation data pattern data
    2- w : represent the weight vector "decision boundary parameters"
    3- Validation_Lables : represent the validation data label 
    4- seq : define the confusion matrix auto naming sequence index
    '''
    
    y=[]
    for i in range(len(w)):
        for j in range(len(all_validation)):
            y_init=np.dot(w[i], all_validation[j])
            y.append(y_init)
    
    y = np.reshape(y,[200,10], order = 'F')
    predict_label = []
    for n in range(len(y)):
        index = np.argmax(y[n])
        predict_label.append(index)
    
    matrix=confusion_matrix(predict_label, Validation_Lables, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Acc = np.trace(matrix)/2
    
    
    ###########################################
    #Plotting Confusion Matrix in better format
    ###########################################
    
    plt.imshow(matrix,cmap=plt.cm.Purples,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix of eta 10^ ' + str(seq))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(Validation_Lables))) # length of classes
    class_labels = ['0','1','2','3','4','5','6','7','8','9']
    tick_marks
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    
    # plotting text value inside cells
    
    thresh = matrix.max() / 2
    for i,j in itertools.product(range(matrix.shape[0]),range(matrix.shape[1])):
        plt.text(j,i,format(matrix[i,j],'d'),horizontalalignment='center',color='white' if matrix[i,j] >thresh else 'black')
    
    # plotting the Matrix output and saving it to JPG format
    Confusionpath = 'E:/Eslam/Data Science/Nile University/CIT-690-A Selected Topics in Machine Learning/Assignment 1 - Perceptron/Validation_confusion Matrices'
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

confusion_all = []
for m in range(len(W)):
    
    Confusion = perceptron_validation(all_validation, W[m], Validation_Lables, seq[m])
    confusion_all.append(Confusion)
    #print(np.shape(confusion_all))

final_confusion = []
for n in range(len(confusion_all)):
    for i in range(len(confusion_all)):
        diag = confusion_all[n][i,i]
        final_confusion.append(diag)
final_confusion = np.reshape(final_confusion, [10,10], order = 'F')        
print(final_confusion)

#####################################################################
#Comuting the best eta index vector to use it to get best accuracy Ws
#####################################################################
flip_confusion = np.flip(final_confusion, axis = 1)
print(flip_confusion)

best_eta_index = []
for l in range(len(flip_confusion)):
    index = (np.argmax(flip_confusion[l]) - 9)*-1
    best_eta_index.append(index)

best_eta_index = np.reshape(best_eta_index, [len(flip_confusion), 1])
os.chdir(Weightpath)  
np.savetxt('w_eta_best.txt', best_eta_index)