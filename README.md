# Dry Beans Classification
![alt text](https://food.unl.edu/newsletters/images/assorted-dry-beans.png)

In this project different machine learning algorithms were used to classify the most well-known 7 types of beans in Turkey; Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira, depending **ONLY** on dimension and shape features of bean varieties with no external discriminatory features.
- This code was part of a Kaggle competition and came in the 9th place out of 81 teams.

MLP, Xgboost, Catboost and LightGBM classifiers were trained and a final VotingClassifier is used resulting F1-score 0.956 on the training data, 0.935 on validation set and 0.938 on the final testing set.  

## Table of contents:
### 1-Dataset
### 2-EDA
### 3-Preprocessing
### 4-Model Training
### 5-Model Training
## Dataset
The dataset provided in this project is obtained from [UC Irvine Machine Learning Repository - Dry Bean Dataset.](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
- **Note**: The data is already splitted with 80% - 20% ratio to training and testing sets respectively, so a part of the data is already separated for final testing and will use the training set for train and validation.

## EDA
- Exploring the dataset, getting summary statistics and checking for null values and duplicates and there weren't any.
- Graphical representations:\
1- Count plot the labels column show the distribution of all classes that showed a slight imbalance but it doesn't affect and no need to handle.\
2- Histogram of numerical features, some distributions have long tails, skewed and most are bi-modal which means that some classes are quite distinct from others.
\
3- Boxplot shows that the "Bombay" & "Horoz" classes are distinct from other classes and that there are some minimal outliers in some features.\
4- The Pearson linear correlation shows that there are lots of highly correlated features.

## Preprocessing
1- Label Encoding the categorical target labels with values from 0 to 6.\
2- Train-Validation split the training dataset with 95% - 5% ratio.\
Now, we have the training set:\
3- Remove outliers from some features with certain threshold.\
4- Feature scaling using StadardScalar()

## Model Training
SVC, MLP, Xgboost, Catboost and LightGBM classifiers were trained on the dataset separately.
- RandomizedSearchCV was used previously in hyperparameter optimization for the models and the best parameters are used directly in this code.
- F1-score and Confusion Matrix  are used to evaluate each model's performance.\
\
Finally, Voting classifier with 'soft' voting is used with the four best models; MLP, Xgboost, Catboost and LightGBM. 

## Setup
To run this project, install:\
1- numpy library\
2- matplotlib library\
3- pandas library\
4- seaborn library\
5- sklearn library\
6- scipy library\
7- os library
