# coding=utf-8

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import time

###################################################################
###################################################################
###################################################################
###################################################################

class plotAnalysis():

###################################################################
###################################################################

    def __init__(self, plotsDir=""):

        sns.set_style("darkgrid")
        colors = ["windows blue", "amber", "greenish", "orange", "sky blue", "greyish", "salmon", "faded green", "lavender", "denim blue", "medium green"]
        # colors = ["windows blue", "aquamarine", "amber", "lightblue", "lavender"]
        custom_palette = sns.xkcd_palette(colors)
        sns.set_palette(custom_palette)
        self.outDir = plotsDir
        
        self.figWidth = 15
        self.figHeight = 8
        self.linewidth = 2

        return

###################################################################
###################################################################

    def plotBarGraph(self, x1, y1, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25, tilt=False, xTickRotation=0, top=20):
        
        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))
            
        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        tempDF = pd.DataFrame({'Country':x1, 'Data':y1})
        tempDF.sort_values("Data", inplace=True, ascending=False)
        tempDF = tempDF.reset_index()
        tempDF = tempDF[:top]

        ax.bar(tempDF['Country'].values, tempDF['Data'], width=1, label=legendLabel1)
        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
        if tilt:
            fig.autofmt_xdate(rotation=xTickRotation)
            
        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotGraph(self, x1, y1, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25, tilt=False, xTickRotation=0, time=True, dateFormat='%Y-%m'):
        
        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))

        if (time==True):
            hfmt = matplotlib.dates.DateFormatter(dateFormat)
            ax.xaxis.set_major_formatter(hfmt)
        
        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        if (time==True):
            ax.plot(x1,y1, label=legendLabel1, lw=self.linewidth)
        else:
            # ax.plot(x1,y1, label=legendLabel1, ls='None', marker='*')
            ax.scatter(x1, y1, label=legendLabel1)


        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
        if tilt:
            fig.autofmt_xdate(rotation=xTickRotation)
            
        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plotPairsDF(self, df, title="", xlabel="", ylabel="", legendLabel1="", outputFileName="", xLabelSize=25):

        fig, ax = plt.subplots(figsize=(self.figWidth, self.figHeight))

        ax.set_title(title, fontsize=xLabelSize)
        ax.set_xlabel(xlabel, fontsize=xLabelSize)
        ax.set_ylabel(ylabel, fontsize=xLabelSize)
        
        # sns.pairplot(df)
        pd.plotting.scatter_matrix(df, marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)

        ax.legend(loc='upper right', prop={'size': xLabelSize-10}, shadow=True, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=xLabelSize)
            
        if len(outputFileName) > 0:
            plt.savefig(self.outDir+outputFileName)

        return

###################################################################
###################################################################

    def plot_corr(self, df, size=10):
        '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot'''

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);

        return

###################################################################
###################################################################