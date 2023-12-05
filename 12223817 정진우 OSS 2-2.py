import pandas as pd
import sklearn as skl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def sort_dataset(dataset_df):
	sortedByYear = dataset_df.sort_values(by='year')
	return sortedByYear
	
def split_dataset(dataset_df):	
    x = dataset_df.drop(columns='salary', axis=1)
    y = dataset_df['salary'] * 0.001
    xTrain = x[:1718]
    xTest = x[1718:]
    yTrain = y[:1718]
    yTest = y[1718:]
    return xTrain,xTest,yTrain,yTest
	

def extract_numerical_cols(dataset_df):
	numerCol = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
	return numerCol
 
def train_predict_decision_tree(X_train, Y_train, X_test):
	dt_cls = DecisionTreeRegressor()
	dt_cls.fit(X_train, Y_train)
	treePred = dt_cls.predict(X_test)
	return treePred
 
def train_predict_random_forest(X_train, Y_train, X_test):
    rf_cls = RandomForestRegressor()
    rf_cls.fit(X_train, Y_train)
    forestPred = rf_cls.predict(X_test)
    return forestPred
	
def train_predict_svm(X_train, Y_train, X_test):
	pipe = make_pipeline(
		StandardScaler(),
		SVR()
	)
	pipe.fit(X_train, Y_train)
	pipePred = pipe.predict(X_test)
	return pipePred
 

def calculate_RMSE(labels, predictions):
	rmseResult = np.sqrt(np.mean((predictions-labels)**2))
	return rmseResult

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
 
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
 
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))