import numpy as np
import pandas as pd
import math

# for data-viz
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


import matplotlib.pyplot as plt
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

from tkinter import Tk
from tkinter.filedialog import askopenfilename


# ----------------------- MAIN EXECUTION STARTS HERE ------------------------------------------- #
print("Choose file with themes and sentiments to get visualizations")
# read the csv file into a dataframe
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')


"""
1. Get the list of most prominent topics in the dataset
"""
df_theme_count = df_reviews['themes'].value_counts().reset_index()
df_theme_count.columns = ['themes','count']

# get total number of unique reviews in the dataset to get % of each theme
total_unique_reviews = df_reviews['Rev_id'].unique().shape[0]
df_theme_count['percent_of_reviews'] = (df_theme_count['count']/total_unique_reviews) * 100

# plot visualization
fig = plt.figure(figsize=(13,13))
plt.gcf().subplots_adjust(bottom=0.08)
sns.set_context("poster")
g = sns.barplot(x="themes", y="percent_of_reviews", data=df_theme_count)
g.set_xticklabels(g.get_xticklabels(), rotation=70)
g.set_xlabel('Themes')
g.set_ylabel('Percentage of Reviews')
plt.savefig('AllThemes.png')
