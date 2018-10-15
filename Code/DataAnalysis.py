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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

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

        print(trainDF.shape)
        print(trainDF.tail(5))

        testDF = self.DFFunc.setupAnalysisDF(testSetupDF)

        train_predictors = trainDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        train_target = pd.DataFrame({'atmosphereCO2':trainDF['atmosphereCO2']})

        test_predictors = testDF.drop(['atmosphereCO2', 'CountryType', 'Country'], axis=1).copy()
        test_target = pd.DataFrame({'atmosphereCO2':testDF['atmosphereCO2']})

        ### Linear Regression

        lr_predictions = self.FitLinearRegression(train_predictors, test_predictors, train_target, test_target)

        ### Decision Tree

        dt_predictions = self.RunDecisionTree(train_predictors, test_predictors, train_target, test_target)

        ### Gradient Boosted Tree

        gbt_predictions = self.RunGradientBoostedTree(train_predictors, test_predictors, train_target, test_target)

        resultsDF = test_target.copy()
        resultsDF['lr_predictions'] = lr_predictions
        resultsDF['lr_res'] = resultsDF['lr_predictions'] - resultsDF['atmosphereCO2']
        resultsDF['lr_res_sqrd'] = resultsDF['lr_res'].apply(lambda x: math.pow(x,2))
        resultsDF['lr_mape'] = resultsDF['lr_res']/resultsDF['atmosphereCO2']

        resultsDF['dt_predictions'] = dt_predictions
        resultsDF['dt_res'] = resultsDF['dt_predictions'] - resultsDF['atmosphereCO2']
        resultsDF['dt_res_sqrd'] = resultsDF['dt_res'].apply(lambda x: math.pow(x,2))
        resultsDF['dt_mape'] = resultsDF['dt_res']/resultsDF['atmosphereCO2']

        resultsDF['gbt_predictions'] = gbt_predictions
        resultsDF['gbt_res'] = resultsDF['gbt_predictions'] - resultsDF['atmosphereCO2']
        resultsDF['gbt_res_sqrd'] = resultsDF['gbt_res'].apply(lambda x: math.pow(x,2))
        resultsDF['gbt_mape'] = resultsDF['gbt_res']/resultsDF['atmosphereCO2']

        print(resultsDF.head(5))
        print(math.pow(resultsDF['lr_res_sqrd'].mean(),0.5))
        print(math.pow(resultsDF['dt_res_sqrd'].mean(),0.5))
        print(math.pow(resultsDF['gbt_res_sqrd'].mean(),0.5))

        # self.plotData.plotResultGraph(resultsDF.index.values, [resultsDF['lr_res'], resultsDF['dt_res'], resultsDF['gbt_res']], title="atmosphereCO2 2014", xlabel="Country", ylabel="atmosphereCO2", legendLabel=["LR_test_Residue", "DT_test_Residue", "GBT_test_Residue"], outputFileName="atmosphereCO2_test_residue.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsDF.index.values, [resultsDF['dt_mape'], resultsDF['gbt_mape']], title="atmosphereCO2 2014", xlabel="Country", ylabel="Percentage", legendLabel=["DT_test_MAPE", "GBT_test_MAPE"], outputFileName="atmosphereCO2_test_MAPE.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsDF.index.values, [resultsDF['dt_mape']], title="atmosphereCO2 2014", xlabel="Country", ylabel="Percentage", legendLabel=["DT_test_MAPE"], outputFileName="atmosphereCO2_test_DT_MAPE.png", tilt=False, xTickRotation=30)

        # self.plotData.plotResultGraph(resultsDF.index.values, [resultsDF['gbt_mape']], title="atmosphereCO2 2014", xlabel="Country", ylabel="Percentage", legendLabel=["GBT_test_MAPE"], outputFileName="atmosphereCO2_test_GBT_MAPE.png", tilt=False, xTickRotation=30)

        # plt.show()

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

        return y_pred

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

        dt_model = dt.fit(train_predictors, train_target.values.ravel())

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


    def RunGradientBoostedTree(self, train_predictors, test_predictors, train_target, test_target):

        # loss=’ls’,  learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001

        print("Running Gradient Boosted Tree.....")

        # Base Model

        base_results, base_modelpred = self.FitGradientBoostedTree(train_predictors, test_predictors, train_target, test_target)

        # Default parameters
        print(base_results)

        # ('subsample', 1.0)
        # ('learning_rate', 0.1)
        # ('min_samples_leaf', 1)
        # ('max_depth', 3)
        # [1, 3, 0.1, 1.0, 47003.42526128258]


        ## Fine tune hyperparameters with Randomized Search

        # Swap for these to running the hyper-parameter search again
        # gbt_max_depth_array = [10,20,30]
        # gbt_min_samples_leaf_array = [1,2,3]
        # gbt_learning_rate_array = [0.05, 0.1, 0.2]
        # gbt_subsample_array = [0.4, 0.8, 1.0]

        # Best parameters
        # 1	10	0.1	1	30726.5481753722
        gbt_max_depth_array = [10]
        gbt_min_samples_leaf_array = [1]
        gbt_learning_rate_array = [0.1]
        gbt_subsample_array = [1.0]

        gbt_results_array = [] # [min_samples_leaf, max_depth, learning_rate, subsample]

        for i in range(0, len(gbt_max_depth_array)):
            for k in range(0, len(gbt_min_samples_leaf_array)):
                for m in range(0, len(gbt_learning_rate_array)):
                    for n in range(0, len(gbt_subsample_array)):
                        gbt_max_depth = gbt_max_depth_array[i]
                        gbt_min_samples_leaf = gbt_min_samples_leaf_array[k]
                        gbt_learning_rate = gbt_learning_rate_array[m]
                        gbt_subsample = gbt_subsample_array[n]

                        gbt_params = {
                            "max_depth":gbt_max_depth,
                            "min_samples_leaf":gbt_min_samples_leaf,
                            "learning_rate":gbt_learning_rate,
                            "subsample":gbt_subsample
                        }

                        gbt_results, gbt_predictions = self.FitGradientBoostedTree(train_predictors, test_predictors, train_target, test_target, gbt_params)
                        gbt_results_array.append(gbt_results)

                        del gbt_max_depth
                        del gbt_min_samples_leaf
                        del gbt_learning_rate
                        del gbt_subsample
                        del gbt_params

        for i in range(0, len(gbt_results_array)):

            print(gbt_results_array[i])

        return gbt_predictions

###################################################################
###################################################################

    def FitGradientBoostedTree(self, train_predictors, test_predictors, train_target, test_target, params={}):

        # loss=’ls’,  learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001

        if bool(params):
            print("Fitting with max_depth = " + str(params["max_depth"]) + ", min_samples_leaf = " + str(params["min_samples_leaf"]) + ", learning_rate = " + str(params["learning_rate"]) + ", subsample = " + str(params["subsample"]) + " ...")
            gbt = ensemble.GradientBoostingRegressor(random_state=42, max_depth=params["max_depth"], min_samples_leaf = params["min_samples_leaf"], learning_rate = params["learning_rate"], subsample = params["subsample"])
        else:
            print("Fitting with default parameters...")
            gbt = ensemble.GradientBoostingRegressor(random_state=42)

        gbt_model = gbt.fit(train_predictors, train_target.values.ravel())

        gbt_rmse, gbt_predictions = self.evaluateModel(model=gbt_model, test_predictors=test_predictors, test_target=test_target, modelName='Gradient Boosted Tree')

        gbt_paramMap = gbt_model.get_params()

        for key in gbt_paramMap.keys():
            # print(key, dt_paramMap[key])

            if key in ['min_samples_leaf']:
                min_samples_leaf = gbt_paramMap[key]
            if key in ['max_depth']:
                max_depth = gbt_paramMap[key]
            if key in ['learning_rate']:
                learning_rate = gbt_paramMap[key]
            if key in ['subsample']:
                subsample = gbt_paramMap[key]
            if bool(params)==False:
                if key in ['min_samples_leaf', 'max_depth', 'learning_rate', 'subsample']:
                    print(key, gbt_paramMap[key])

        return [min_samples_leaf, max_depth, learning_rate, subsample, gbt_rmse], gbt_predictions

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

if __name__ == "__main__":

    startTime = time.time()
    
    dataDir = '../Data/'
    plotsDir = '../Plots/'
        
    Analysis_Object = DataAnalysis(dataDir, plotsDir)
    
    Analysis_Object.runAnalysis()
    
    endTime = time.time()
    
    print ("Time elapsed: " + repr(endTime-startTime))