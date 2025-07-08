# ValvePredictionProject
Predicting percentage of valve opening, depending upon the liquid level. 
# Predicting valve position on the basis of liquid level
In this notebook, we will be trying to predict the position, the regulatory valve should assume on the basis of liquid level
## 1. Problem definition
> How well can the valve position be predicted
## 2. Data
> We have two sets of data, the dataset for valve position as well as the liquid level
* The valve position dataset provides the valve position in terms of percentage (0-100)
* The liquid level dataset provides liquid level in the tank in mililitres (ml)
* We are to combine these two separate datasets in order to form our training, validation and test datasets
### 2.1. How to get the train, validation and test data sets
* Firstly we will be combining the valve position and liquid level data from the `Процесс 2_Уровень_008.KIP1.L_S1_2_month.csv` and `Процесс_2_Положение_клпапна_008_KIP1_Pos_KlR7_2_1month.csv` data sets
* Then we will be using the train_test_split to split the combined data in the ratio of 75:25 to get our training and validation datasets
* As for the test data, we will use `Процесс_3_Уровень` data set as the test data set
## 3. Evaluation and Improvisation
We will be using RMSLE (Root Mean Squared Log Error) as well as other evaluation methods. 
The goal of the regression metrics will be to minimize the error.
