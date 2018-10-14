# coding=utf-8
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta, time, date

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

###################################################################
###################################################################
###################################################################
###################################################################

class FitFunctions():

###################################################################
###################################################################

    def __init__(self):
        

        return
        
###################################################################
###################################################################

    def evaluate(self, model, test_features, test_labels):
        
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f}.'.format(np.mean(errors)))
        print('MAPE: {:0.4f}.'.format(mape))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        
        return accuracy

###################################################################
###################################################################

    def FitRandomForest(self, train_predictors, train_target, paramGrid={}, searchType = "Random"):
        
        if (bool(paramGrid)==False):

            rf_model = RandomForestRegressor(random_state=42)

        else:

            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            rf = RandomForestRegressor(random_state=42)

            if (searchType=="Random"):

                # Random search of parameters, using 3 fold cross validation, 
                # search across 100 different combinations, and use all available cores
                print ('Performing Random Search of hyper parameters for Random Forest...')
                rf_model = RandomizedSearchCV(estimator = rf, param_distributions = paramGrid)#, n_iter = 100, cv = 3, verbose=1, n_jobs = -1)

            else:
                print ('Performing Grid Search of hyper parameters for Random Forest...')
                # # Instantiate the grid search model
                rf_model = GridSearchCV(estimator = rf, param_grid = paramGrid)#, cv = 3, n_jobs = -1, verbose = 1)

        # Fit the random search model
        rf_model.fit(train_predictors, train_target.values.ravel())

        if (bool(paramGrid)==True):

            print("Model best parameters:")
            print(rf_model.best_params_)
        
        return rf_model

###################################################################
###################################################################

    def FitGradientBoostedTree(self, train_predictors, train_target, paramGrid={}, searchType = "Random"):
        
        if (bool(paramGrid)==False):

            gbt_model = GradientBoostingRegressor(random_state=42)

        else:

            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            gbt = GradientBoostingRegressor(random_state=42)

            if (searchType=="Random"):

                # Random search of parameters, using 3 fold cross validation, 
                # search across 100 different combinations, and use all available cores
                print ('Performing Random Search of hyper parameters for Gradient Boosted Tree...')
                gbt_model = RandomizedSearchCV(estimator = gbt, param_distributions = paramGrid, n_iter = 100, cv = 3, verbose=1, n_jobs = -1)

            else:

                print ('Performing Grid Search of hyper parameters for Random Forest...')
                # # Instantiate the grid search model
                gbt_model = GridSearchCV(estimator = gbt, param_grid = paramGrid, cv = 3, n_jobs = -1, verbose = 1)

        # Fit the random search model
        gbt_model.fit(train_predictors, train_target.values.ravel())

        if (bool(paramGrid)==True):

            print("Model best parameters:")
            print(gbt_model.best_params_)
        
        return gbt_model

###################################################################
###################################################################