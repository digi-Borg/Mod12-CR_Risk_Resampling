# Module 12: Credit Risk Analysis Report

*"A FinTech project as a 'Credit Risk Analyst' identifying credit worthiness of loans."*

## Overview of the Analysis

This project constructs a program to identify the credit worthiness of borrowers at a peer to peer lending services company using an historical dataset of lending activity. Observing that the orignal data has risks of a quantity imbalance between the data sets in its classification of 'healthy loans' to 'high-risk-loans', an unbiased comparison needs to be made with a resampled data model to predict 'high-risk-loans' more accurately. 

In order to fairly predict with a higher degree of accuracy without bias, a process of three stages are followed: model-fit-predict. This process or pattern allows an algorithm to mold and adjust itself to new input data and predict future outcomes. Employing python and pandas functions with **imbalanced learn** and **sklearn** libraries, specific tools are imported from them to fit data into logistic regression models. These models appropiate the data into majority and minority class labels to make predictions more accurate in identifying high-risk-loans. 

Using the `sklearn` library, a `LogisticRegression` module is imported to create and train two datasets, the original and resampled data. Each dataset is fitted into separate models in order to compare new predicted data outputs for identifying 'high-risk-loans' more accurately. First, the original dataset is modeled for predictions. Next, from the imbalanced-learn(`imblearn`) library importing the `RandomOverSampler` module is used for resampling the minority data label or 'high-risk-loans'. 

For both cases, we compared the `value_counts` of the target variable 'loan status' before fitting the model in a `LogisticRegression` classifier. After making a prediction of the models, a `balanced_accuracy_score` is calculated to summarize the model predictions; a `confusion_matrix` is generated to compare accurate predictions with inaccurate predictions to evaluate the model's performance; finally a `classification_report` is generated to give a more detailed view of the models preformance for closer examination of precision and sensitivity.


--- 

## Results

The following describes the balanced accuracy scores, the precision and recall scores of both machine learning models.

* Machine Learning Model 1 (Original Unbalanced):
  * Model 1 Accuracy: Is found to be at 0.95205 or 95.21%
  * M1 Precision: The precision is perfect for the 0 class (1.00) or 100%, but lower class 1 (0.85) or 85%.     Where '0' represents a 'healthy loan' and, '1' represents a high-risk-loan for the scores. 
  * M1 Recall score: for sensitivity is near perfect for '0' class at 99%, while at 91% for '1' class or high-risk-loans.



* Machine Learning Model 2 (Resampled):
  * M2 Accuracy: Is at 0.99367 or 99.37%
  * M2 Precision: Is perfect 100% for the 'healthy loans' and is slightly lower at 84% for high-risk-loans.
  * M2 Recall scores: 'Healthy loans' matches the original model results at 99%, but high-risk-loans are higher matching the 99% in this resampled model. 

--- 

## Summary

Summarizing the results of the two MLearning models is that both are perfectly close in calculating healthy loans with precision, however there are slight differences in the resampled data: 

* My 1st observation is that the accuracy score of the resampled data is higher than the original data (0.9936 vs 0.9521) defining its superior precision  at identifying 'true positives' and 'true negatives'. With a closer observation in the classification report of the 'original' model displaying a slightly better metric at identifying minority class 'high-risk-loans' (0.85 vs 0.84) of  the 'resample', but after looking at the 'resampled' data 'recall' with its sensitivity metric being much higher at (0.99 vs 0.91) than the 'original' this indicates that the 'high-risk-loans' should be lower. 

* Does performance depend on the problem we are trying to solve? The performance of both models are very close and precision are high for each.  The 'resampled' data model is close, but has an edge in correctly classifying a higher percentage of high-risk-loans. The (RandomOversample) model using 'resampled' data is slightly better at identifying high-risk-loans that are likely to default than the 'original' model having an imbalanced dataset.

However considering that 'Precision' on both were almost identical, another run or two with a resampled data model is strongly considered to find other results to clarify any questionably close metric discrepancies.

---
--- 

## Technologies

The software operates on python 3.9 with the installation package imports embedded with Anaconda3 installation. Using the `sklearn` library, a `LogisticRegression` module is imported to create and train two datasets. First, the original dataset is modeled for predictions and using the imbalanced-learn(`imblearn`) library the `RandomOverSampler` module is used for resampling the minority data label.  The tools that you need for this module also include, `classification_report_imbalanced` from `imblearn` library, `balanced_accuracy_score` and `confusion_matrix` are imported from `sklearn` library. The `pydot plus` provides a Python Interface to Graphviz Dot language used to creat and visualize decision trees. 


* [anaconda3](https://docs.anaconda.com/anaconda/install/windows/e) 

* [imblearn](https://imbalanced-learn.org/stable/) 

*  [sklearn](https://scikit-learn.org/stable/install.html) 

* [pydot plus](https://pypi.org/project/pydotplus/) .

--- 

## Installation Guide

Before running the applications first install the following libraries: 

```python libraries
!pip install -U imbalanced-learn
!pip install -U scikit-learn
!pip install pydotplus
```
```from pathlib import Path
import pandas as pd 
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced 
``` 
---
```
python in Jupyter Notebook:
credit_risk_resampling.ipynb
```

---

## Contributors

*Provided to you by digi-Borg FinTek*, 
Dana Hayes: nydane1@gmail.com

---

## License

Columbia U. Engineering