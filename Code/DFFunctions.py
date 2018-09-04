# coding=utf-8
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta, time, date

###################################################################
###################################################################
###################################################################
###################################################################

class DFFunctions():

###################################################################
###################################################################

    def __init__(self):
        
        self.countryTypeDict = eval(open("countryTypeDict.txt").read())

        return
        
###################################################################
###################################################################

    def getCSVDF(self, csv, skiprow=0, filter=False):

        ## This function loads the csv files into dataframes, and converts the values in the respective time columns into datetime formats

        ## lists timeCol and timeFormat must have the same length

        outDF = pd.read_csv(csv, skiprows=skiprow)

        if (filter==True):
            outDF = self.getCountryTypeCol(outDF)
            outDF = outDF[outDF['CountryType']=='Country']
            outDF = outDF.reset_index()
        
        return outDF

###################################################################
###################################################################

    def getCountryTypeCol(self, inDF):

        ## This function loads the csv files into dataframes, and converts the values in the respective time columns into datetime formats

        outDF = inDF.copy()

        outDF['CountryType'] = outDF['Country Name'].map(self.countryTypeDict)

        return outDF

###################################################################
###################################################################