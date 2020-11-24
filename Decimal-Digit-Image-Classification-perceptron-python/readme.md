# Decimal-Digit-Image-Classification-perceptron-python "Coding & Machine Learning"

## Introduction 
1. In the first part of this project we will design a Perceptron-based classification algorithm that can recognize scanned images of the 10 digits (0 to 9) provided in the file “Assignment 1 Dataset.zip”. The zip file contains three folders: “Train”, “Validation” and “Test”. The “Train” folder contains 240 images for each digit, while each of the “Validation” and “Test” folders contain 20 images for each digit. 

The images in the “Train” folder should be used to train a classifier for each digit. The folder contains a file named “Training Labels.txt” which includes the labels of the 2400 images in order. You need to train the classifiers using each of the following values for the learning rate η = 1, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9. For all Perceptrons, use an initial weight vector that has 1 as the first component (w1) and the rest are zeros. After the classifiers are trained, test each classifier using the images given in the “Test” folder.

2. In the second part of the project Use the data in the “Validation” folder to find the value of η that achieves the best accuracy for each digit classifier. Use the best classifier of each digit to classify the data in the “Test” folder. The “Validation” folder also contains a text file named “Validation Labels.txt” which include the labels of the 200 images in order.

## Repository contents
The code for this project is divided into 3 .py files one for training the classifier, another one for applying the model on testing data and get the accuracy for different η and the last part use validation data to get the best eta for each digit classifier then apply the best weight vectors again on the test data and output the accuracy.

The repository includes 
**1. Perceptron-based image classifier training .py file** this is the code for training the data.   
**2. Perceptron-based image classifier testing .py file** this is the code for applying the model we got from training in testing data and get accuracy for different η.  
**3. Perceptron-based image classifier valid .py file**this is the code for applynig the model in validation data then get for each classfier η with the best performance and after that we apply the new model in testing data and get accuracy.  
**4. Confusion_matrices_images zipped file** and it inludes 10 images that represents 10 confusion matrices. the first 9 matrices named “Confusion-x.jpg”, where x is absolute value of the power of 10 of η and the last matrix named "Confusion_b.jpg" and it represents confusion matrix with best accuracy after applying the model in validation data, get best eta per classifier then apply best model on testing data.  
**5. Dataset zipped file** and it inludes training, testing and validation datasets.   

**Please note that this project can be simpler If we made some dimensionality reduction but the main goal of the project is to apply perceptron concept on high dimensional data.**
