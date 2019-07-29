# Match-Winner-Prediction-Using-Python

***Introduction***

In this project we implemented a 3-level decision tree to predict the outcome of the games Liverpool played in the 2017/2018 premier league season. The training data given in the file “Training_Data.xlsx” is the outcome of all games of all other teams that ended in the win of one of the two competing teams. The decision tree should predict, based on the values of the given attributes, whether the home team or the away team of the game in which Liverpool is playing will win the game. The attributes given for each game are as follows:

H: Home Team  
A: Away Team  
HS: Home Team Shots  
AS: Away Team Shots  
HST: Home Team Shots on Target  
AST: Away Team Shots on Target  
HF: Home Team Fouls Committed  
AF: Away Team Fouls Committed  
HC: Home Team Corners  
AC: Away Team Corners  
HY: Home Team Yellow Cards  
AY: Away Team Yellow Cards  
HR: Home Team Red Cards  
AR: Away Team Red Cards  
  
***Implementation steps and results***  

Given that the data is numeric, we descritized the root node of the tree so as to have 2 possible values. The nodes in the two next levels have 3 possible values. For discretization of the root node, it was based on whether the value is above or below the mean value of the attribute. For the nodes of the other two levels we used equal spacing discretization.

When we applied the implemented algorithm on Liverpool matches we predicted whether home team or away team won with 88.46% accuracy which shows how crucial statistics are in the world of sport and how using machine learning can affect the strategies of coaches and as a result matches results.

***Repository contents***

**The repository includes**

**Decision_Tree_Training_model.py** file this is the Training Model code for the project.  
**Decision_Tree_Test_model.py** file this is the Test Model code for the project.
**confusion Matrix.png** it is a confusion matrix that shows the algorithm accuracy when applied to Liverpool data(Testing data)   
**Training_Data.xlsx** file this is the data used for training.   
**Liverpool.xlsx file** this is the data used for testing the accuracy of Decisin tree algorithm implemented.

