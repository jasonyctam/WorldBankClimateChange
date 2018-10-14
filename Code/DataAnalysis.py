# coding=utf-8
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import DFFunctions as DFFuncs
import FitFunctions as FitFuncs
import plotAnalysis as plotMethods
import datetime as dt
import numpy as np
import ast # For converting string to dictionary

# from sklearn import datasets, linear_model
# import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


###################################################################
###################################################################
###################################################################
###################################################################

class DataAnalysis():

###################################################################
###################################################################

    def __init__(self, dataDir, plotsDir):

        self.dataDir = dataDir
        self.plotsDir = plotsDir

        self.figWidth = 15
        self.figHeight = 8
        self.linewidth = 2

        self.tiltBool = True
        self.rotation = 30

        self.plotData = plotMethods.plotAnalysis(plotsDir=self.plotsDir)
        self.DFFunc = DFFuncs.DFFunctions()
        self.FitFunc = FitFuncs.FitFunctions()

        return

###################################################################
###################################################################
    
    def runAnalysis(self):

        self.landForest_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_AG.LND.FRST.ZS_DS2_en_csv_v2_10052112.csv', skiprow=4)
        self.atmosphereCO2_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_EN.ATM.CO2E.KT_DS2_en_csv_v2_10051706.csv', skiprow=4)
        self.GDP_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10051799.csv', skiprow=4)
        self.populationTotal_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.POP.TOTL_DS2_en_csv_v2_10058048.csv', skiprow=4)
        self.populationUrban_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_10034507.csv', skiprow=4)

        # print(self.landForest_DF.tail(5))
        # print(self.atmosphereCO2_DF.tail(5))
        # print(self.GDP_DF.tail(5))
        # print(self.populationTotal_DF.tail(5))
        # print(self.populationUrban_DF.tail(5))

        # print(self.populationUrban_DF.shape)

        dfList = [self.landForest_DF, self.atmosphereCO2_DF, self.GDP_DF, self.populationTotal_DF, self.populationUrban_DF]
        dfColumnHeaders = ['landForest', 'atmosphereCO2', 'GDP', 'populationTotal', 'populationUrban']

        trainSetupDF = pd.DataFrame({'Country':self.landForest_DF['Country Name'], 'CountryType':self.landForest_DF['CountryType']})
        testSetupDF = pd.DataFrame({'Country':self.landForest_DF['Country Name'], 'CountryType':self.landForest_DF['CountryType']})

        for i in range(0,len(dfList)):
            tempDF = dfList[i]
            # Pick year with data in every variable, particularly atmosphereCO2
            trainSetupDF[dfColumnHeaders[i]] = tempDF['2013']
            testSetupDF[dfColumnHeaders[i]] = tempDF['2014']

        trainDF = self.DFFunc.setupAnalysisDF(trainSetupDF)
        # trainDF = self.DFFunc.setupAnalysisDF(trainSetupDF, countryType='Group')
        print(trainDF.shape)
        print(trainDF.tail(5))

        testDF = self.DFFunc.setupAnalysisDF(testSetupDF)

        train_predictors = trainDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        train_target = pd.DataFrame({'atmosphereCO2':trainDF['atmosphereCO2']})

        test_predictors = testDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        test_target = pd.DataFrame({'atmosphereCO2':testDF['atmosphereCO2']})

        ### Linear Regression

        # self.FitLinearRegression(train_predictors, test_predictors, train_target, test_target)

        ### Random Forest

        # self.FitRandomForest(train_predictors, test_predictors, train_target, test_target)

        # rf = best_grid

        ### Decision Tree
        dt_predictions = self.RunDecisionTree(train_predictors, test_predictors, train_target, test_target)

        ### Gradient Boosted Tree

        # self.FitGradientBoostedTree(train_predictors, test_predictors, train_target, test_target)

        resultsDF = test_target.copy()
        resultsDF['dt_predictions'] = dt_predictions
        resultsDF['dt_res'] = resultsDF['dt_predictions'] - resultsDF['atmosphereCO2']
        resultsDF['dt_res_sqrd'] = resultsDF['dt_res'].apply(lambda x: math.pow(x,2))
        print(resultsDF.head(5))
        print(math.pow(resultsDF['dt_res_sqrd'].mean(),0.5))


 

        return

###################################################################
###################################################################

    def FitLinearRegression(self, train_predictors, test_predictors, train_target, test_target):

        regressor = LinearRegression()
        regressor.fit(train_predictors, train_target)

        print('Intercept: \n', regressor.intercept_)
        print('Coefficients: \n', regressor.coef_)

        y_pred = regressor.predict(test_predictors)
        print('Linear Regression R squared": %.4f' % regressor.score(test_predictors, test_target))

        lin_mse = mean_squared_error(y_pred, test_target)
        lin_rmse = np.sqrt(lin_mse)
        print('Linear Regression RMSE: %.4f' % lin_rmse)

        lin_mae = mean_absolute_error(y_pred, test_target)
        print('Linear Regression MAE: %.4f' % lin_mae)

        return

###################################################################
###################################################################

    def RunDecisionTree(self, train_predictors, test_predictors, train_target, test_target):

        print("Running Decision Tree.....")

        # Base Model

        base_results, base_modelpred = self.FitDecisionTree(train_predictors, test_predictors, train_target, test_target)

        # Default parameters
        print(base_results)

        # Swap for these to running the hyper-parameter search again
        # dt_max_depth_array = [10,20,30]
        # dt_max_leaf_nodes_array = [32,34,36]
        # dt_min_samples_leaf_array = [1,2,3]

        # Best Parameters
        # 1	10	36	32136.4225470143
        dt_max_depth_array = [10]
        dt_max_leaf_nodes_array = [36]
        dt_min_samples_leaf_array = [1]

        dt_results_array = [] # minInstancesPerNode, maxDepth, maxBins, rmse

        for i in range(0, len(dt_max_depth_array)):
            for j in range(0, len(dt_max_leaf_nodes_array)):
                for k in range(0, len(dt_min_samples_leaf_array)):
                    dt_max_depth = dt_max_depth_array[i]
                    dt_max_leaf_nodes = dt_max_leaf_nodes_array[j]
                    dt_min_samples_leaf = dt_min_samples_leaf_array[k]

                    dt_params = {
                        "max_depth":dt_max_depth,
                        "max_leaf_nodes":dt_max_leaf_nodes,
                        "min_samples_leaf":dt_min_samples_leaf
                    }

                    dt_results, dt_predictions = self.FitDecisionTree(train_predictors, test_predictors, train_target, test_target, dt_params)

                    dt_results_array.append(dt_results)

                    del dt_max_depth
                    del dt_max_leaf_nodes
                    del dt_min_samples_leaf
                    del dt_params
                    del dt_results

        for i in range(0, len(dt_results_array)):

            print(dt_results_array[i])

        return dt_predictions

###################################################################
###################################################################

    def FitDecisionTree(self, train_predictors, test_predictors, train_target, test_target, params={}):

        if bool(params):
            print("Fitting with max_depth = " + str(params["max_depth"]) + ", max_leaf_nodes = " + str(params["max_leaf_nodes"]) + ", min_samples_leaf = " + str(params["min_samples_leaf"]) + " ...")
            dt = DecisionTreeRegressor(random_state=42, max_depth=params["max_depth"], max_leaf_nodes = params["max_leaf_nodes"], min_samples_leaf = params["min_samples_leaf"])
        else:
            print("Fitting with default parameters...")
            dt = DecisionTreeRegressor(random_state=42)


        dt_model = dt.fit(train_predictors, train_target)

        dt_rmse, dt_predictions = self.evaluateModel(model=dt_model, test_predictors=test_predictors, test_target=test_target, modelName='Decision Tree')

        dt_paramMap = dt_model.get_params()

        for key in dt_paramMap.keys():
            # print(key, dt_paramMap[key])

            if key in ['min_samples_leaf']:
                min_samples_leaf = dt_paramMap[key]
            if key in ['max_depth']:
                max_depth = dt_paramMap[key]
            if key in ['max_leaf_nodes']:
                max_leaf_nodes = dt_paramMap[key]
            if bool(params)==False:
                if key in ['min_samples_leaf', 'max_depth', 'max_leaf_nodes']:
                    print(key, dt_paramMap[key])

        # print("Decision Tree Root Mean Squared Error (RMSE) on test data = %g" % dt_rmse)

        return [min_samples_leaf, max_depth, max_leaf_nodes, dt_rmse], dt_predictions

###################################################################
###################################################################

    def FitGradientBoostedTree(self, train_predictors, test_predictors, train_target, test_target):

        # loss=’ls’,  learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001

        ## Fine tune hyperparameters with Randomized Search

        # Swap for these to running the hyper-parameter search again
        # gbt_maxDepth_array = [10,20,30]
        # gbt_minInstancesPerNode_array = [1,2,3]
        # gbt_maxIter_array = [20, 25, 30]
        # gbt_stepSize_array = [0.05, 0.1, 0.2]
        # gbt_subsamplingRate_array = [0.4, 0.8, 1.0]

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 3, num = 3)]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [2, 4, 6, 8, 10, 12]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4, 5]


        # Create the random grid
        gbt_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

        # Base Model

        base_model = self.FitFunc.FitGradientBoostedTree(train_predictors=train_predictors, train_target=train_target)
        base_accuracy = self.FitFunc.evaluate(base_model, test_predictors, test_target.values.ravel())

        ## Random Search
        gbt_random = self.FitFunc.FitRandomForest(paramGrid=gbt_grid, train_predictors=train_predictors, train_target=train_target)
        best_random = gbt_random.best_estimator_
        random_accuracy = self.FitFunc.evaluate(best_random, test_predictors, test_target.values.ravel())
        print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

        model = ensemble.GradientBoostingRegressor()
        model.fit(train_predictors, train_target)

        print('Gradient Boosting R squared": %.4f' % model.score(test_predictors, test_target))

        y_pred = model.predict(test_predictors)
        model_mse = mean_squared_error(y_pred, test_target)
        model_rmse = np.sqrt(model_mse)
        print('Gradient Boosting RMSE: %.4f' % model_rmse)

        feature_labels = np.array(['landForest', 'GDP', 'populationTotal', 'populationUrban'])
        importance = model.feature_importances_
        feature_indexes_by_importance = importance.argsort()
        for index in feature_indexes_by_importance:
            print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))

        return

###################################################################
###################################################################

    def evaluateModel(self, model, test_predictors, test_target, modelName=''):

        y_pred = model.predict(test_predictors)
        mse = mean_squared_error(y_pred, test_target)
        rmse = np.sqrt(mse)
        # print(modelName + ' RMSE: %.4f' % rmse)

        return float(rmse), y_pred

###################################################################
###################################################################

    def FitRandomForest(self, train_predictors, test_predictors, train_target, test_target):

        # forest_reg = RandomForestRegressor(random_state=42)
        # forest_reg.fit(train_predictors, train_target)

        # print('Random Forest R squared": %.4f' % forest_reg.score(test_predictors, test_target))

        # y_pred = forest_reg.predict(test_predictors)
        # forest_mse = mean_squared_error(y_pred, test_target)
        # forest_rmse = np.sqrt(forest_mse)
        # print('Random Forest RMSE: %.4f' % forest_rmse)

        # print("Model best parameters:")
        # print(forest_reg.best_params_)

        ## Fine tune hyperparameters with Randomized Search

        # # Number of features to consider at every split
        # max_features = ['auto', 'sqrt']

        # # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        # max_depth.append(None)

        # # Minimum number of samples required to split a node
        # min_samples_split = [2, 4, 6, 8, 10, 12]

        # # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 3, 4, 5]

        # # Method of selecting samples for training each tree
        # bootstrap = [True, False]

        # # Create the random grid
        # random_grid = {'n_estimators': n_estimators,
        #             'max_features': max_features,
        #             'max_depth': max_depth,
        #             'min_samples_split': min_samples_split,
        #             'min_samples_leaf': min_samples_leaf,
        #             'bootstrap': bootstrap}

        # Base Params
        # ('bootstrap', True)
        # ('min_samples_leaf', 1)
        # ('n_estimators', 10)
        # ('min_samples_split', 2)

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 5, stop = 50, num = 10)]

        # Minimum number of samples required to split a node
        min_samples_split = [2, 4, 6, 8, 10, 12]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4, 5]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Base Model

        base_model = self.FitFunc.FitRandomForest(train_predictors=train_predictors, train_target=train_target)
        base_accuracy = self.FitFunc.evaluate(base_model, test_predictors, test_target.values.ravel())
        # base_params = ast.literal_eval(base_model.get_params())
        base_params = base_model.get_params()
        print('Parameters for the base model:')
        for key in base_params.keys():
            print(key, base_params[key])

        ## Random Search
        rf_random = self.FitFunc.FitRandomForest(paramGrid=random_grid, train_predictors=train_predictors, train_target=train_target)
        best_random = rf_random.best_estimator_
        random_accuracy = self.FitFunc.evaluate(best_random, test_predictors, test_target.values.ravel())
        print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

        # Model best parameters:
        # {'bootstrap': False, 'min_samples_leaf': 1, 'n_estimators': 1400, 'max_features': 'sqrt', 'min_samples_split': 10, 'max_depth': 50}
        # Model Performance
        # Average Error: 49455.5544.
        # Accuracy = -177.44%.
        # Improvement of -510.95%.

        # Model best parameters:
        # {'bootstrap': False, 'min_samples_leaf': 2, 'n_estimators': 60, 'max_features': 'sqrt', 'min_samples_split': 12, 'max_depth': 10}
        # Model Performance
        # Average Error: 60940.7022.
        # Accuracy = -240.68%.
        # Improvement of -657.42%.



        ## Fine tune hyperparameters with Grid Search

        # param_grid = {
        #     'n_estimators': [400, 600, 800, 1000, 1200],
        #     'max_features': ['auto'],
        #     'min_samples_split': [8, 10, 12],
        #     'min_samples_leaf': [1, 2, 3],
        #     'max_depth': [60, 70, 80, 90, 100],
        #     'bootstrap': [False]
        # }

        # Create the param grid
        param_grid = {'n_estimators': n_estimators,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        grid_search = self.FitFunc.FitRandomForest(paramGrid=param_grid, train_predictors=train_predictors, train_target=train_target, searchType="Grid")
        best_grid = grid_search.best_estimator_
        grid_accuracy = self.FitFunc.evaluate(best_grid, test_predictors, test_target.values.ravel())

        print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

        # Model best parameters:
        # {'bootstrap': False, 'min_samples_leaf': 2, 'n_estimators': 400, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 60}
        # Model Performance
        # Average Error: 65963.5829.
        # Accuracy = 43.37%.
        # Improvement of 0.44%.
        # Time elapsed: 571.5303890705109


        return


###################################################################
###################################################################


if __name__ == "__main__":

    startTime = time.time()
    
    dataDir = '../Data/'
    plotsDir = '../Plots/'
        
    Analysis_Object = DataAnalysis(dataDir, plotsDir)
    
    Analysis_Object.runAnalysis()
    
    endTime = time.time()
    
    print ("Time elapsed: " + repr(endTime-startTime))