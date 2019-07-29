from math import log2, log
import pandas as pd
import numpy as np
import os

#%%
# =============================================================================
# Entropy function
# =============================================================================

def entropy(data_list):
    '''
    return the Entropy of a probability distribution:
    entropy(p) = − SUM (data_list * log(data_list) )
    defintion:
            entropy is a metric to measure the uncertainty of a probability distribution.
    entropy ranges between 0 to 1
    Low entropy means the distribution varies (peaks and valleys).
    High entropy means the distribution is uniform.
    See:
            http://www.cs.csi.cuny.edu/~imberman/ai/Entropy%20and%20Information%20Gain.htm
    '''

    total = 0
    if sum(data_list) == 0:
        
        total = 0
        
    else:
        for p in data_list:
            p = p / sum(data_list)
            if p != 0:
                total += p * log(p, 2)
            else:
                total += 0
        total *= -1
    return total

#%%
# =============================================================================
# function to compute gain for L2
# =============================================================================
def get_branch_node(dataset, nodedf, branchdf, Es, branchlevel):
    columnnames=list(nodedf.columns)
    high_A=0
    medium_A=0
    low_A=0
    high_H=0
    medium_H=0
    low_H=0
    high_acc=0
    med_acc=0
    low_acc=0
    Gain1=[]
    for i in columnnames[:-1]:
        high_A=0
        medium_A=0
        low_A=0
        high_H=0
        medium_H=0
        low_H=0
        high_acc=0
        med_acc=0
        low_acc=0
        for j in range(len(branchdf)):
            q1= ((dataset[i].max()-dataset[i].min())/3)+dataset[i].min()
            q2= ((dataset[i].max()-dataset[i].min())*2/3)+dataset[i].min()
            
            if (branchdf[i][j] > q2):
                high_acc +=1
                if (branchdf['FTR'][j] == 'H'):
                    high_H += 1
                else:
                    high_A +=1
            if (branchdf[i][j] > q1) and (branchdf[i][j] <= q2):
                med_acc += 1
                if (branchdf['FTR'][j] == 'H'):
                    medium_H +=1
                else:
                    medium_A +=1
            if (branchdf[i][j] <= q1):
                low_acc +=1 
                if (branchdf['FTR'][j] == 'H'):
                    low_H +=1
                else:
                    low_A +=1
       
        low=[low_A, low_H]
        med=[medium_A, medium_H]
        high=[high_A, high_H]
                
        outputs=[low, med, high]
        '''for l in range(len(outputs)):
            if outputs[l] == 0:
                outputs[l]=sys.float_info.min
        [low_A,low_H, medium_A, medium_H, high_A,high_H]=outputs'''
        #print(str(i) + ' : ' + str(outputs))
        E_high=entropy(high)
        E_medium=entropy(med)
        E_low=entropy(low)
        gain_i=Es-((((high_H+high_A)/len(branchdf))*E_high) + (((medium_H+medium_A)/len(branchdf))*E_medium) +(((low_H+low_A)/len(branchdf))*E_low))
        Gain1.append(gain_i)
    max_gain_index=np.argmax(Gain1)
    #print('\n Gains are:  \n' + str(Gain1) + '\n')
    print('\n'+ str(branchlevel) + ' node is : ' + str(columnnames[max_gain_index]) + '\n' + 'with Gain = ' + str(max(Gain1)))
    print('maximum Gain index = ' + str(np.argmax(Gain1)))
    nodename=columnnames[max_gain_index]
    return max_gain_index

#%%
os.getcwd()

train=pd.read_excel('Training_Data.xlsx', header = 0) #import excel training set file
train=train.drop(train.columns[0:2], axis=1)#remove first two columns (team names)
traindf=pd.DataFrame(train) #this is dataframe
train_mins=list(traindf.min())#result all colums minimum
train_means=list(traindf.mean())#result all colums means
train_maxs=list(traindf.max())#result all colums maximum
print('dataset shape =  ' + str(traindf.shape))

# =============================================================================
# computing root node
# =============================================================================
column_names=list(traindf.columns)
high_A=0
low_A=0
high_H=0
low_H=0
As=sum(traindf['FTR']=='A')
Hs=sum(traindf['FTR']=='H')
'''Entropy_s=(-As/len(traindf))*log2(As/len(traindf))-(Hs/len(traindf))*log2(Hs/len(traindf))'''
Entropy_s=entropy([As,Hs])
Gain=[]
for i in column_names[:-1]:
    high_A=0
    low_A=0
    high_H=0
    low_H=0
    for j in range(len(traindf)):
        if (traindf[i][j] > traindf[i].mean()) and (traindf['FTR'][j] == 'H'):
            high_H +=1
        elif (traindf[i][j] > traindf[i].mean()) and (traindf['FTR'][j] == 'A'):
            high_A +=1
        elif (traindf[i][j] <= traindf[i].mean()) and (traindf['FTR'][j] == 'H'):
            low_H +=1   
        else:
            low_A +=1
    high=[high_A, high_H]
    low=[low_A, low_H]
    E_high=entropy(high)
    E_low=entropy(low)
    gain_i=Entropy_s-((((high_H+high_A)/len(traindf))*E_high) + (((low_H+low_A)/len(traindf))*E_low))
    Gain.append(gain_i)
max_gain_index=np.argmax(Gain)
#print(column_names)
#print('Gains are:  \n' + str(Gain) + '\n')
print('\nroot node is : ' + str(column_names[max_gain_index]) + '\n' + 'with Gain = ' + str(max(Gain)))
print('maximum Gain index = ' + str(np.argmax(Gain)))
# =============================================================================
# creating new 2 DF based on root node HST (high & low) discritization
#to compute second leaf
# =============================================================================
traindf['HST_level']=['high'  if traindf['HST'][i] > traindf['HST'].mean() else 'low' for i in range(len(traindf))]
HST_mean=traindf['HST'].mean()
level1df=traindf.drop(traindf.columns[np.argmax(Gain)], axis=1)
level1df=level1df.set_index(['HST_level'])
#subsetting the original dataset to two datasets based on root node(high/low)
l1_high=level1df.loc['high']
l1_high=l1_high.reset_index()
l1_low=level1df.loc['low']
l1_low=l1_low.reset_index()
#computing node set entropy
highs=len(l1_high) #number of HST with high category > mean
lows=len(l1_low) #number of HST with low category <= mean
A_high=sum(l1_high['FTR']=='A') #number of FTR=A and HST=high
H_high=sum(l1_high['FTR']=='H') #number of FTR=H and HST=high
A_low=sum(l1_low['FTR']=='A') #number of FTR=A and HST=low
H_low=sum(l1_low['FTR']=='H') #number of FTR=H and HST=low

print('\nHST descritization point -mean- = ' + str(traindf['HST'].mean()))
print('HST High Branch size =   ' + str(l1_high.shape) + '\n\t Home = ' + str(H_high) + '\n\t Away = ' + str(A_high))
print('HST Low Branch size =    ' + str(l1_low.shape) + '\n\t Home = ' + str(H_low) + '\n\t Away = ' + str(A_low))

E_1high=entropy([A_high, H_high])
E_1low=entropy([A_low, H_low])

# =============================================================================
# computing level 1 nodes (HST High branch)
# =============================================================================
Gain1= get_branch_node(traindf, level1df, l1_high, E_1high, 'HST-HIGH')
Gain2= get_branch_node(traindf, level1df, l1_low, E_1low, 'HST-LOW')

# =============================================================================
# creating new 3 DF based on 1st split node HY and HST = High
# =============================================================================
HY_q1= ((traindf['HY'].max()-traindf['HY'].min())/3)+traindf['HY'].min()
HY_q2= ((traindf['HY'].max()-traindf['HY'].min())*2/3)+traindf['HY'].min()
l1_high['HY_level']=['low'  if l1_high['HY'][i] <= HY_q1 else 'medium' if (l1_high['HY'][i] > HY_q1 and l1_high['HY'][i] <= HY_q2) else 'high' for i in range(len(l1_high))]
l1_high.drop('HST_level', axis=1, inplace=True)
level2df=l1_high.drop(['HY', 'AST'], axis=1)
level2df=level2df.set_index(['HY_level'])
#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
HY_h_high=level2df.loc['high']
HY_h_high=HY_h_high.reset_index()
HY_h_med=level2df.loc['medium']
HY_h_med=HY_h_med.reset_index()
HY_h_low=level2df.loc['low']
HY_h_low=HY_h_low.reset_index()
#computing node set entropy
A_high=sum(HY_h_high['FTR']=='A') #number of FTR=A and HST=low & HY=high
H_high=sum(HY_h_high['FTR']=='H') #number of FTR=H and HST=low & HY=high
A_med=sum(HY_h_med['FTR']=='A') #number of FTR=A and HST=high & HY=med
H_med=sum(HY_h_med['FTR']=='H') #number of FTR=H and HST=high & HY=med
A_low=sum(HY_h_low['FTR']=='A') #number of FTR=A and HST=low & HY=low
H_low=sum(HY_h_low['FTR']=='H') #number of FTR=H and HST=low & HY=low

E_HY_h_high=entropy([A_high, H_high])
E_HY_h_med=entropy([A_med, H_med])
E_HY_h_low=entropy([A_low, H_low])

print('\nHY descritization points (Q1, Q2)= [' + str(HY_q1) + ', ' + str(HY_q2) + ']')
print('HST High Branch & HY low size =    ' + str(HY_h_low.shape) + '\n\t Home = ' + str(H_low) + '\n\t Away = ' + str(A_low))
print('HST High Branch & HY  medium size =   ' + str(HY_h_med.shape) + '\n\t Home = ' + str(H_med) + '\n\t Away = ' + str(A_med))
print('HST High Branch & HY  high size =   ' + str(HY_h_high.shape) + '\n\t Home = ' + str(H_high) + '\n\t Away = ' + str(A_high))

# =============================================================================
#creating new 3 DF based on 1st split node AST and HST = LOW
# =============================================================================
AST_q1= ((traindf['AST'].max()-traindf['AST'].min())/3)+traindf['AST'].min()
AST_q2= ((traindf['AST'].max()-traindf['AST'].min())*2/3)+traindf['AST'].min()
l1_low['AST_level']=['low'  if l1_low['AST'][i] <= AST_q1 else 'medium' if (l1_low['AST'][i] > AST_q1 and l1_low['AST'][i] <= AST_q2) else 'high' for i in range(len(l1_low))]
l1_low.drop('HST_level', axis=1, inplace=True)
level2df_l=l1_low.drop(['HY', 'AST'], axis=1)
level2df_l=level2df_l.set_index(['AST_level'])

#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
AST_l_high=level2df_l.loc['high']
AST_l_high=AST_l_high.reset_index()
AST_l_med=level2df_l.loc['medium']
AST_l_med=AST_l_med.reset_index()
AST_l_low=level2df_l.loc['low']
AST_l_low=AST_l_low.reset_index()
# =============================================================================
# computing node set entropy
# =============================================================================
A_high=sum(AST_l_high['FTR']=='A') #number of FTR=A and HST=low & AST=med
H_high=sum(AST_l_high['FTR']=='H') #number of FTR=H and HST=low & AST=med
A_med=sum(AST_l_med['FTR']=='A') #number of FTR=A and HST=low & AST=med
H_med=sum(AST_l_med['FTR']=='H') #number of FTR=H and HST=low & AST=med
A_low=sum(AST_l_low['FTR']=='A') #number of FTR=A and HST=low & AST=low
H_low=sum(AST_l_low['FTR']=='H') #number of FTR=H and HST=low & AST=low
E_AST_l_high=entropy([A_high, H_high])
E_AST_l_med=entropy([A_med, H_med])
E_AST_l_low=entropy([A_low, H_low])

print('\nAST descritization points (Q1, Q2)= [' + str(AST_q1) + ', ' + str(AST_q2) + ']')
print('HST LOW Branch & AST low size =    ' + str(AST_l_low.shape) + '\n\t Home = ' + str(H_low) + '\n\t Away = ' + str(A_low))
print('HST LOW Branch & AST  medium size =   ' + str(AST_l_med.shape) + '\n\t Home = ' + str(H_med) + '\n\t Away = ' + str(A_med))
print('HST LOW Branch & AST  high size =   '+ str(AST_l_high.shape) + '\n\t Home = ' + str(H_high) + '\n\t Away = ' + str(A_high))


#%%
# =============================================================================
# HST=high, AST=low (AST_h-low)
# =============================================================================

get_branch_node(traindf, level2df, HY_h_low,E_HY_h_low, 'HST-High -- HY-Low')
get_branch_node(traindf, level2df, HY_h_med, E_HY_h_med, 'HST-High -- HY-Med')
get_branch_node(traindf, level2df, HY_h_high, E_HY_h_high, 'HST-High -- HY-High')
get_branch_node(traindf, level2df_l, AST_l_low, E_AST_l_low,'HST-low -- AST-low')
get_branch_node(traindf, level2df_l, AST_l_med, E_AST_l_med, 'HST-low -- AST-Med')

# =============================================================================
# create subsets for HST=high, HY =low branch to do majority voting for classification
# =============================================================================
AC_q1= ((traindf['AC'].max()-traindf['AC'].min())/3)+traindf['AC'].min()
AC_q2= ((traindf['AC'].max()-traindf['AC'].min())*2/3)+traindf['AC'].min()
HY_h_low['AC_level']=['low'  if HY_h_low['AC'][i] <= AC_q1 else 'medium' if (HY_h_low['AC'][i] > AC_q1 and HY_h_low['AC'][i] <= AC_q2) else 'high' for i in range(len(HY_h_low))]
HY_h_low.drop('HY_level', axis=1, inplace=True)
AC_df=HY_h_low.drop(['HF', 'AC', 'AY'], axis=1)
AC_df=AC_df.set_index(['AC_level'])
#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
AC_h_high=AC_df.loc['high']
AC_h_high=AC_h_high.reset_index()
AC_h_med=AC_df.loc['medium']
AC_h_med=AC_h_med.reset_index()
AC_h_low=AC_df.loc['low']
AC_h_low=AC_h_low.reset_index()

A_high=sum(AC_h_high['FTR']=='A') #number of FTR=A and HST=high & HY=high
H_high=sum(AC_h_high['FTR']=='H') #number of FTR=H and HST=high & HY=high
A_med=sum(AC_h_med['FTR']=='A') #number of FTR=A and HST=high & HY=med
H_med=sum(AC_h_med['FTR']=='H') #number of FTR=H and HST=high & HY=med
A_low=sum(AC_h_low['FTR']=='A') #number of FTR=A and HST=low & HY=low
H_low=sum(AC_h_low['FTR']=='H') #number of FTR=H and HST=low & HY=low

print('\nAC descritization points (Q1, Q2)= [' + str(AC_q1) + ', ' + str(AC_q2) + ']')
print('HST High Branch & HY low & AC low size =    ' + str(AC_h_low.shape) + '\n\t Home = ' + str(H_low) + '\n\t Away = ' + str(A_low))
print('HST High Branch & HY low & AC med size =   ' + str(AC_h_med.shape) + '\n\t Home = ' + str(H_med) + '\n\t Away = ' + str(A_med))
print('HST High Branch & HY low & AC high size =   ' + str(AC_h_high.shape) + '\n\t Home = ' + str(H_high) + '\n\t Away = ' + str(A_high))

#%% Middle HF branch subsetting
# =============================================================================
# create subsets for HST=high, HY =med branch to do majority voting for classification
# =============================================================================
HF_q1= ((traindf['HF'].max()-traindf['HF'].min())/3)+traindf['HF'].min()
HF_q2= ((traindf['HF'].max()-traindf['HF'].min())*2/3)+traindf['HF'].min()
HY_h_med['HF_level']=['low'  if HY_h_med['HF'][i] <= HF_q1 else 'medium' if (HY_h_med['HF'][i] > HF_q1 and HY_h_med['HF'][i] <= HF_q2) else 'high' for i in range(len(HY_h_med))]
HY_h_med.drop('HY_level', axis=1, inplace=True)
HF_df=HY_h_med.drop(['HF', 'AC', 'AY'], axis=1)
HF_df=HF_df.set_index(['HF_level'])
#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
HF_h_high=HF_df.loc['high']
HF_h_high=HF_h_high.reset_index()
HF_h_med=HF_df.loc['medium']
HF_h_med=HF_h_med.reset_index()
HF_h_low=HF_df.loc['low']

l = ('Home = 1 \n\t Away = 0')
m= HF_h_med['FTR'].value_counts()
h= HF_h_high['FTR'].value_counts()
print('\nHF descritization points (Q1, Q2)= [' + str(HF_q1) + ', ' + str(HF_q2) + ']')
print('HST High Branch & HY med & HF low size =    ' + str(HF_h_low.shape) + '\n\t ' + str(l))
print('HST High Branch & HY med & HF med size =   ' + str(HF_h_med.shape) + '\n\t Home = ' + str(m['H']) + '\n\t Away = ' + str(m['A']))
print('HST High Branch & HY med & HF high size =   ' + str(HF_h_high.shape) + '\n\t Home = ' + str(h['H']) + '\n\t Away = ' + str(h['A']))

#%% High HF branch subsetting
# =============================================================================
# create subsets for HST=high, HY =High branch to do majority voting for classification
# =============================================================================
AY_q1= ((traindf['AY'].max()-traindf['AY'].min())/3)+traindf['AY'].min()
AY_q2= ((traindf['AY'].max()-traindf['AY'].min())*2/3)+traindf['AY'].min()
HY_h_high['AY_level']=['low'  if HY_h_high['AY'][i] <= AY_q1 else 'medium' if (HY_h_high['AY'][i] > AY_q1 and HY_h_high['AY'][i] <= AY_q2) else 'high' for i in range(len(HY_h_high))]
HY_h_high.drop('HY_level', axis=1, inplace=True)
AY_df=HY_h_high.drop(['HF', 'AC', 'AY'], axis=1)
AY_df=AY_df.set_index(['AY_level'])
#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
AY_h_med=AY_df.loc['medium']
AY_h_med=AY_h_med.reset_index()
AY_h_low=AY_df.loc['low']
AY_h_low=AY_h_low.reset_index()

l = ('Home = 1 \n\t Away = 0')
m= AY_h_med['FTR'].value_counts()

print('\nAY descritization points (Q1, Q2)= [' + str(AY_q1) + ', ' + str(AY_q2) + ']')
print('HST High Branch & HY high & AY low size =    ' + str(AY_h_low.shape) + '\n\t ' + str(l))
print('HST High Branch & HY high & AY med size =   ' + str(AY_h_med.shape) + '\n\t Home = 0 \n\t Away = ' + str(m['A']))
print('HST High Branch & HY high & AY high size =  0 ')


#%% Low branch subsetting
# =============================================================================
# create subsets for HST=high, AST =low branch to do majority voting for classification
# =============================================================================
AY_q1= ((traindf['AY'].max()-traindf['AY'].min())/3)+traindf['AY'].min()
AY_q2= ((traindf['AY'].max()-traindf['AY'].min())*2/3)+traindf['AY'].min()
AST_l_low['AY_level']=['low'  if AST_l_low['AY'][i] <= AY_q1 else 'medium' if (AST_l_low['AY'][i] > AY_q1 and AST_l_low['AY'][i] <= AY_q2) else 'high' for i in range(len(AST_l_low))]

AST_l_low.drop('AST_level', axis=1, inplace=True)

AY_df=AST_l_low.drop(['AY', 'AF'], axis=1)
AY_df=AY_df.set_index(['AY_level'])

#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
AY_h_high=AY_df.loc['high']
AY_h_high=AY_h_high.reset_index()
AY_h_med=AY_df.loc['medium']
AY_h_med=AY_h_med.reset_index()
AY_h_low=AY_df.loc['low']
AY_h_low=AY_h_low.reset_index()

l = AY_h_low['FTR'].value_counts()
m = AY_h_med['FTR'].value_counts()
h = AY_h_high['FTR'].value_counts()
print('\nAY descritization points (Q1, Q2)= [' + str(AY_q1) + ', ' + str(AY_q2) + ']')
print('HST low Branch & AST low & AY low size =    ' + str(AY_h_low.shape) + '\n\t Home = '+ str(l['H']) + '\n\t Away = ' + str(l['A']))
print('HST low Branch & AST low & AY med size =   ' + str(AY_h_med.shape) + '\n\t Home = ' + str(m['H']) + ' \n\t Away = ' + str(m['A']))
print('HST low Branch & AST low & AY high size =  ' + str(AY_h_high.shape) + '\n\t Home = ' + str(h['H']) + ' \n\t Away = 0' )

#%% Med branch subsetting
# =============================================================================
# create subsets for HST=high, AST =med branch to do majority voting for classification
# =============================================================================
AF_q1= ((traindf['AF'].max()-traindf['AF'].min())/3)+traindf['AF'].min()
AF_q2= ((traindf['AF'].max()-traindf['AF'].min())*2/3)+traindf['AF'].min()
AST_l_med['AF_level']=['low'  if AST_l_med['AF'][i] <= AF_q1 else 'medium' if (AST_l_med['AF'][i] > AF_q1 and AST_l_med['AF'][i] <= AF_q2) else 'high' for i in range(len(AST_l_med))]

AST_l_med.drop('AST_level', axis=1, inplace=True)
AF_df=AST_l_med.drop(['AY', 'AF'], axis=1)
AF_df=AF_df.set_index(['AF_level'])

#subsetting the original dataset to two datasets based on root node(high/low)
# =============================================================================
AF_h_high=AF_df.loc['high']
AF_h_high=AF_h_high.reset_index()
AF_h_med=AF_df.loc['medium']
AF_h_med=AF_h_med.reset_index()
AF_h_low=AF_df.loc['low']
AF_h_low=AF_h_low.reset_index()

l = AF_h_low['FTR'].value_counts()
m = AF_h_med['FTR'].value_counts()
h = AF_h_high['FTR'].value_counts()
print('\nAF descritization points (Q1, Q2)= [' + str(AF_q1) + ', ' + str(AF_q2) + ']')
print('HST low Branch & AST med & AF low size =    ' + str(AF_h_low.shape) + '\n\t Home = '+ str(l['H']) + '\n\t Away = ' + str(l['A']))
print('HST low Branch & AST med & AF med size =   ' + str(AF_h_med.shape) + '\n\t Home = ' + str(m['H']) + ' \n\t Away = ' + str(m['A']))
print('HST low Branch & AST med & AF high size =  ' + str(AF_h_high.shape) + '\n\t Home = 0' '\n\t Away = ' + str(h['A']) )

