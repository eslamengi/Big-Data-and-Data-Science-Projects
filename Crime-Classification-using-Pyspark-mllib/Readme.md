
# <center>Crime Classification using Pyspark "Big Data"

## Project Overview:

From 1934 to 1963, San Francisco was infamous for housing some of the world's most notorious criminals on the inescapable island of Alcatraz.

Today, the city is known more for its tech scene than its criminal past. But, with rising wealth inequality, housing shortages, and a proliferation of expensive digital toys riding BART to work, there is no scarcity of crime in the city by the bay.

From Sunset to SOMA, and Marina to Excelsior, this competition's dataset provides nearly 12 years of crime reports from across all of San Francisco's neighborhoods. 

**Target:** Given time and location, need to predict the category of crime that occurred, only by using the machine learnig library (mllib) in the Pyspark.sql


**Competition Link on Kaggle:**  
https://www.kaggle.com/c/sf-crime


## Dataset Description:

This dataset contains incidents derived from SFPD Crime Incident Reporting system. The data ranges from 1/1/2003 to 5/13/2015. The training set and test set rotate every week, meaning week 1,3,5,7... belong to test set, week 2,4,6,8 belong to training set. 

**Data link:**  
https://www.kaggle.com/c/sf-crime/data


**Data fields:**  

Dates - timestamp of the crime incident  
Category - category of the crime incident (only in train.csv). This is the target variable you are going to predict.  
Descript - detailed description of the crime incident (only in train.csv)  
DayOfWeek - the day of the week  
PdDistrict - name of the Police Department District  
Resolution - how the crime incident was resolved (only in train.csv)  
Address - the approximate street address of the crime incident   
X - Longitude  
Y - Latitude  

## Evaluation Matrix:  
evaluated using the multi-class logarithmic loss. Each incident has been labeled with one true class. For each incident, you must submit a set of predicted probabilities (one for every class). The formula is then,

![image.png](attachment:image.png)

where N is the number of cases in the test set, M is the number of class labels, log is the natural logarithm, yij is 1 if observation i is in class j and 0 otherwise, and pij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given incident are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with max(min(p,1−10−15),10−15).

## steps:
1. Some Exploratory Data analysis and visualization  
2. Training 3 different Classification models and each model I modified 1 hyperparameter each run as below  
    a. randomforest numtrees = 10  
    b. randomforest numtrees = 20  
    c. randomforest numtrees = 30  
    d. Logisticregression regparam = 1  
    e. Logisticregression regparam = 0.1  
    f. Logisticregression regparam = 0.01  
    g. decisiontree maxdepth = 2  
    h. decisiontree maxdepth = 5  
    i. decisiontree maxdepth = 10   
3. prediction result for each model was then uploaded to kaggle to use the Logloss evaluation matrix to know the best score   
    

## Uploads:  
1- Project code  
2- best score presentation  
3- project html file showing the code and run results  
4- Readme


