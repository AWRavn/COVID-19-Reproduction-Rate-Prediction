from prepare_data import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mlxtend.evaluate import bias_variance_decomp


RAND = 43
CORES = 12

class Trees:

	def __init__(self, X_train, X_test, y_train, y_test, model_name):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		self.maxdepth = 4
		self.n_estimators = 500
		self.model_name = model_name


	def fit(self, plot=True, save=True, printout=True, default=False):

		# Create empty arrays
		mse = np.zeros(self.maxdepth)
		r2 = np.zeros(self.maxdepth)
		mae = np.zeros(self.maxdepth)
		rmse = np.zeros(self.maxdepth)

		# Establish model and prediction
		model = self.apply_model(model=self.model_name, default=default)
		model.fit(self.X_train, self.y_train)
		y_pred = model.predict(self.X_test)

		# Populate data arrays
		mse = mean_squared_error(self.y_test, y_pred)
		r2 = r2_score(self.y_test, y_pred)
		mae = mean_absolute_error(self.y_test, y_pred)
		rmse = mean_squared_error(self.y_test, y_pred, squared=False)

		# Print result
		if printout==True:
			print('MSE Error:', mse)
			print('R2:', r2)
			print('MAE Error:', mae)
			print('RMSE Error:', rmse)

		# Plot the results
		if plot==True:
			plt.figure()
			p1 = max(max(y_pred), max(self.y_test))
			p2 = min(min(y_pred), min(self.y_test))
			plt.scatter(self.y_test, y_pred, color="orange")
			plt.plot([p1, p2], [p1, p2], color="black", label="target", linewidth=2)
			plt.xlabel("data")
			plt.ylabel("target")
			plt.axis('equal')
			plt.savefig('figs/prediction.png')
			#plt.show()

		return (mse, r2, mae, rmse)


	def apply_model(self, model=None, default=False):


		if model=='gradient_boosting':
			if default==True:
				model = GradientBoostingRegressor(random_state=RAND)
			else:
				model = GradientBoostingRegressor(
											max_depth=12, 
											subsample=0.75,
											n_estimators=2000, 
											learning_rate=0.01,
											max_features='auto',
											random_state=RAND
											)

		elif model=='XGBoost':
			if default==True:
				model = xgb.XGBRegressor(random_state=RAND)
			else:
				model =  xgb.XGBRegressor(
											objective ='reg:squarederror',
											max_depth=5, 
											subsample=0.5,
											colsample_bytree=0.3,
											n_estimators=1000, 
											learning_rate=0.1,
											max_features='auto',
											random_state=RAND
											)

		elif model=='random_forest_regression':
			if default==True:
				model = RandomForestRegressor(random_state=RAND)
			else:
				model = RandomForestRegressor(
											n_estimators=2000,
											max_depth=100,
											min_samples_split=2,
											max_features='auto',
											oob_score=True,
											bootstrap=True,
											#n_jobs=3,
											random_state=RAND
											)
		return model


	def get_importance_scores(self):

		model = self.model_name
		model = self.apply_model(model=self.model_name, default=True)
		model.fit(self.X_train, self.y_train)

		if self.model_name=='XGBoost':
			model.get_booster().get_score(importance_type='weight')
			scores = model.feature_importances_
		else:
			scores = model.feature_importances_

		return scores


	def get_permutation_based_importance_scores(self):

		model = self.model_name
		model = self.apply_model(model=self.model_name, default=True)
		model.fit(self.X_train, self.y_train)

		scores = permutation_importance(model, self.X_test, self.y_test)

		return scores


	def tune_parameters(self, type='grid'):

		model = self.apply_model(model=self.model_name, default=True)

		if self.model_name=='XGBoost':
			params = { 
					'max_depth': [4, 5, 6, 8, 10],
					'subsample': [0.5], #[0.5, 0.75, 1]
					'colsample_bytree': [0.3, 0.5, 0.7], #[0.7]
					'n_estimators': [100, 500, 1000, 1500],
					'learning_rate': [0.01, 0.1, 0.2, 0.3],
					}
		elif self.model_name=='gradient_boosting':
			params = { 
					'max_depth': [4, 5, 6, 8, 10, 12, 15, 20],
					'subsample': [0.5, 0.75, 1],
					'n_estimators': [100, 500, 1000, 1500, 2000],
					'learning_rate': [0.01, 0.1, 0.2, 0.3],
					}
		elif self.model_name=='random_forest_regression':
			params = { 
					'max_depth': [10, 50, 100, 150],
					'min_samples_split': [2, 3],
					'n_estimators': [100, 500, 800, 1000, 1200, 1500, 2000],
					'bootstrap': [True],
					'n_jobs': [-1],
					}

		if type=='grid':
			clf = GridSearchCV(
							estimator=model,
							param_grid=params,
							scoring='neg_mean_squared_error',
							#verbose=4,
							n_jobs=-1,
							cv=3
							)

		elif type=='random':
			clf = RandomizedSearchCV(
							estimator=model,
							param_distributions=params,
							scoring='neg_mean_squared_error',
							#verbose=4,
							n_jobs=-1,
							cv=3,
							n_iter=600,
							#pre_dispatch=10,
							)

		clf.fit(self.X_train, self.y_train)

		return clf.best_params_


def analyse_feature_importances(dataset, name):

	# Train test split
	X_train, X_test, y_train, y_test = split_and_scale_data(dataset)

	# Create empty arrays
	cols = list(dataset)
	cols.pop() # remove reprodction_rate column
	rfr = np.zeros(len(cols))
	gb = np.zeros(len(cols))
	xgb = np.zeros(len(cols))

	# Get data
	for a in ['XGBoost', 'gradient_boosting', 'random_forest_regression']:
		# Get data
		model = Trees(X_train, X_test, y_train, y_test, a)
		row = model.get_importance_scores()
		row = row.importances_mean

		if a=='XGBoost': xgb = row
		elif a=='gradient_boosting': gb = row
		elif a=='random_forest_regression': rfr = row

	scores = pd.DataFrame(
						data=[rfr, gb, xgb],
						index=["Random Forest Regression", "Gradient Boosting", "XGBoost"],
						columns=cols
						)
	
	# Save scores
	if save==True:
		scores.to_csv('data/feat_scores_2_{}.csv'.format(name))

	return scores


def analyse_feature_permutation_importances(dataset, name, save=False):

	# Train test split
	X_train, X_test, y_train, y_test = split_and_scale_data(dataset)

	# Create empty arrays
	cols = list(dataset)
	cols.pop() # remove reprodction_rate column
	rfr = np.zeros(len(cols))
	gb = np.zeros(len(cols))
	xgb = np.zeros(len(cols))

	for a in ['XGBoost', 'gradient_boosting', 'random_forest_regression']:
		# Get data
		model = Trees(X_train, X_test, y_train, y_test, a)
		row = model.get_permutation_based_importance_scores()
		row = row.importances_mean

		if a=='XGBoost': xgb = row
		elif a=='gradient_boosting': gb = row
		elif a=='random_forest_regression': rfr = row

	scores = pd.DataFrame(
						data=[rfr, gb, xgb],
						index=["Random Forest Regression", "Gradient Boosting", "XGBoost"],
						columns=cols
						)
	
	# Save scores
	if save==True:
		scores.to_csv('data/feat_scores_2_{}.csv'.format(name))

	return scores


def split_and_scale_data(dataset):

	X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]

	# Train test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RAND)

	# Scale data
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	return X_train_scaled, X_test_scaled, y_train, y_test




def main():

	countries= ["sweden"] #["norway", "sweden", "usa"]

	for country in countries:

		print("Evaluating parameters for {}...".format(country))

		##### Load and pre-process the data
		# df = get_dataframe() # Uncomment to create processed data
		df = pd.read_csv('data/processed_covid_data_{}.csv'.format(country))

		#### Find feature importance scores and print the values
		#scores1 = analyse_feature_importances(dataset=df, name=country)

		#### Find feature permutation based importance and print the values
		#scores2 = analyse_feature_permutation_importances(dataset=df, name=country)

		"""
		# Selected features
		df = df[["total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million", 
				"new_tests", "total_tests", "positive_rate", "tests_per_case",
				"people_fully_vaccinated", "reproduction_rate"]]
		"""

		
		# Train test split
		X_train, X_test, y_train, y_test = split_and_scale_data(dataset=df)


		"""
		methods = ['gradient_boosting', 'random_forest_regression'] 
	
		### Find hyperparameters RFR and GB
		for method in methods:
			print("Getting {} hyperparameters....".format(method))
			model = Trees(X_train, X_test, y_train, y_test, method)
			params = model.tune_parameters(type='grid')
			print("Best parameters:", params)
		"""
		"""
		### Find hyperparameters XGBOOST
		for method in methods:
			print("Getting {} hyperparameters....".format('XGBoost'))
			model = Trees(X_train, X_test, y_train, y_test, 'XGBoost')
			params = model.tune_parameters(type='random')
			print("Best parameters:", params)
		"""
		
	
		
		#### Get performance metrics
		methods = ['gradient_boosting'] #['gradient_boosting', 'random_forest_regression', 'XGBoost'] 
		for method in methods:
			print("Getting {} data....".format(method))
			model = Trees(X_train, X_test, y_train, y_test, method)
			scores = model.fit(default=False, printout=True) # default=False for optimized parameters
		

		


if __name__ == "__main__":
	main()