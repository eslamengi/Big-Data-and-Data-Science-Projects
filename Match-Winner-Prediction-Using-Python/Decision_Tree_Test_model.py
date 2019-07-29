import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt


train=pd.read_excel('Training_Data.xlsx', header = 0) #import excel training set file
train=train.drop(train.columns[0:2], axis=1)#remove first two columns (team names)
traindf=pd.DataFrame(train) #this is dataframe

#decision points based on training dataset
HST = [traindf['HST'].mean()]
HY = [((traindf['HY'].max()-traindf['HY'].min())/3)+traindf['HY'].min(), ((traindf['HY'].max()-traindf['HY'].min())*2/3)+traindf['HY'].min()]
AST = [((traindf['AST'].max()-traindf['AST'].min())/3)+traindf['AST'].min(), ((traindf['AST'].max()-traindf['AST'].min())*2/3)+traindf['AST'].min()]
AC = [((traindf['AC'].max()-traindf['AC'].min())/3)+traindf['AC'].min(), ((traindf['AC'].max()-traindf['AC'].min())*2/3)+traindf['AC'].min()]
HF = [((traindf['HF'].max()-traindf['HF'].min())/3)+traindf['HF'].min(), ((traindf['HF'].max()-traindf['HF'].min())*2/3)+traindf['HF'].min()]
AY = [((traindf['AY'].max()-traindf['AY'].min())/3)+traindf['AY'].min(), ((traindf['AY'].max()-traindf['AY'].min())*2/3)+traindf['AY'].min()]
AF = [((traindf['AF'].max()-traindf['AF'].min())/3)+traindf['AF'].min(), ((traindf['AF'].max()-traindf['AF'].min())*2/3)+traindf['AF'].min()]

test = pd.read_excel('Liverpool.xlsx', header = 0) #import excel training set file
test = test.drop(test.columns[0:2], axis=1)#remove first two columns (team names)
test = pd.DataFrame(test) #this is dataframe
columnnames = test.columns
actual_FTR = list(test['FTR'])

predicted_FTR=[]

# =============================================================================
# Classification Conditions
# =============================================================================
for r in range(len(test)):
    if (test['HST'][r] > HST and test['HY'][r] <= HY[0]):
        predicted_FTR.append('H')
    elif (test['HST'][r] > HST and (test['HY'][r] > HY[0] and test['HY'][r] <= HY[1]) and test['HF'][r] <= HF[1]):
        predicted_FTR.append('H')
    elif (test['HST'][r] > HST and test['HY'][r] > HY[1] and test['AY'][r] <= AY[0]):
        predicted_FTR.append('H')
    elif (test['HST'][r] <= HST and test['AST'][r] <= AST[0] and (test['AY'][r] <= AY[0] or test['AY'][r] > AY[1])):
        predicted_FTR.append('H')
    else :
        predicted_FTR.append('A')


correct=0
for i in range(len(predicted_FTR)):
    if predicted_FTR[i] == actual_FTR[i]:
       correct +=1
    
        
accuracy=correct * 100/len(actual_FTR)
print('Accuracy =   ' + str(accuracy) + '  %')

# =============================================================================
# Compute and plotting confusion Matrix
# =============================================================================
matrix=confusion_matrix(predicted_FTR, actual_FTR, labels=['H', 'A'])
print('Confusion Matris:\n'+ str(matrix))
plt.imshow(matrix,cmap=plt.cm.Reds,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix of Liverpool test data ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(len(set(actual_FTR))) # length of classes
class_labels = ['H', 'A']
tick_marks
plt.xticks(tick_marks,class_labels)
plt.yticks(tick_marks,class_labels)

