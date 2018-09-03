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