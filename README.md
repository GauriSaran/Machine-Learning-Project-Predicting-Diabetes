# Machine-Learning-Project-Predicting-Diabetes
Predicting Onset of Diabetes using Classification Algorithms
# Predicting Diabetes using Classification
### Machine Learning Project

In this project, onset of diabetes is predicted based on different diagnostics measures. The dataset used is the Pima Indians Diabetes Database taken from kaggle. All All the patients in the dataset are females atleast 21 years of age and of Pima Indian heritage.

The different variables used in the prediction of the outcome are:

* Pregnancies: Number of times pregnant
* Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance
* Blood Pressure: Diastolic blood pressure (mmHg)
* Skin Thickness: Triceps skin fold thickness
* Insulin: 2-Hour serum insulin(mu/Uml)
* BMI: Body mass index (weight in kg/(height in m)^2)
* DiabetesPedigreeFunction: Diabetes pedigree function
* Age: Age (years)

Different classification algorithms are used in the prediction of diabetes and are evaluated based on two different evaluation metrics:

    f1-score value which is the harmonic average of both precision and recall of a model.
    Jaccard similarity score which shows hows close are the predicted labels to actual labels.

Precision shows percentage of values that were True out of all the values predicted. Recalls shows percentage of values that were True out of all the values that are actually True. 

F1-score is the harmonic average of both Precision and recall. 

For the best accuracy of a model both Precision and Accuracy have to be high to have a high F-1 score. Higher the F1-score, the better the model and higher the Jaccard similarity score, the better is the model. The following are the machine learning classification algorithms/ models which will be used to train the data for finding the right parameters for prediction of onset of diabetes: 

* Support Vector Machines
* K-Nearest Neighbors
* Decision Trees
* Logistic Regression

## Data
Data used has been downloaded from https://www.kaggle.com/uciml/pima-indians-diabetes-database The file is named diabetes.csv

## Programming Language
Python 3.6 has been used for this project.

## Libraries
Following libraries have been used in this project:
* numpy
* pandas
* matplotlib
* seaborn
* sklearn

**Statistical Methodologies Employed**: 
* Data Wrangling
* Exploratory Analysis
* Predictive Analysis
* Support Vector Machine
* K-Nearest Neighbors
* Decision Trees
* Logistic Regression

## Notebook
Python notebook file is available with this project and is named: Diabetes_ML_Classification.ipynb

## Powerpoint Summary
diabetes_Classification.pptx
