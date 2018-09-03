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