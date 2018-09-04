# coding=utf-8
import time
import pandas as pd
import matplotlib.pyplot as plt
import DFFunctions as DFFuncs
import plotAnalysis as plotMethods
import datetime as dt

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

        return

###################################################################
###################################################################
    
    def runAnalysis(self):

        self.landForest_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_AG.LND.FRST.ZS_DS2_en_csv_v2_10052112.csv', skiprow=4, filter=True)
        self.atmosphereCO2_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_EN.ATM.CO2E.KT_DS2_en_csv_v2_10051706.csv', skiprow=4, filter=True)
        self.GDP_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10051799.csv', skiprow=4, filter=True)
        self.populationTotal_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.POP.TOTL_DS2_en_csv_v2_10058048.csv', skiprow=4, filter=True)
        self.populationUrban_DF = self.DFFunc.getCSVDF(csv=self.dataDir+'API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_10034507.csv', skiprow=4, filter=True)

        print(self.landForest_DF.tail(5))
        print(self.atmosphereCO2_DF.tail(5))
        print(self.GDP_DF.tail(5))
        print(self.populationTotal_DF.tail(5))
        print(self.populationUrban_DF.tail(5))

        print(self.populationUrban_DF.shape)

        dfList = [self.landForest_DF, self.atmosphereCO2_DF, self.GDP_DF, self.populationTotal_DF, self.populationUrban_DF]
        dfColumnHeaders = ['landForest', 'atmosphereCO2', 'GDP', 'populationTotal', 'populationUrban']

        combineDF = pd.DataFrame({'Country':self.landForest_DF['Country Name']})

        for i in range(0,len(dfList)):
            tempDF = dfList[i]
            # Pick year with data in every variable, particularly atmosphereCO2
            combineDF[dfColumnHeaders[i]] = tempDF['2014']

        # print(combineDF.head(5))
        combineDF = combineDF.dropna()
        combineDF = combineDF.reset_index()
        combineDF = combineDF.drop('index', axis=1)
        combineDF = combineDF.set_index('Country')
        print(combineDF)
        # self.plotData.plot_corr(combineDF)

        # print(len(self.landForest_DF['Country Name'].unique()))
        # print(len(self.atmosphereCO2_DF['Country Name'].unique()))
        # print(len(self.GDP_DF['Country Name'].unique()))
        # print(len(self.populationTotal_DF['Country Name'].unique()))
        # print(len(self.populationUrban_DF['Country Name'].unique()))

        # print(self.landForest_DF['2017'])

        self.plotData.plotPairsDF(combineDF)

        # self.plotData.plotGraph(combineDF['landForest'], combineDF['atmosphereCO2'], title="atmosphereCO2 VS landForest", xlabel="landForest", ylabel="atmosphereCO2", legendLabel1="Countries", outputFileName="atmosphereCO2_VS_landForest.png", time=False)

        # self.plotData.plotBarGraph(self.GDP_DF['Country Name'], self.GDP_DF['2017'], title="GDP 2017", xlabel="Country", ylabel="GDP", legendLabel1="GDP 2017", outputFileName="GDP_2017.png", tilt=self.tiltBool, xTickRotation=self.rotation)

        plt.show()

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