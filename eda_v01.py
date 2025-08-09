# load data
import pandas as pd
import numpy as np
from statistics import mean
# eda
import matplotlib.pyplot as plt
import seaborn as sns
# pre-processing
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#============================== EDA ==============================
'''
Details  of 'data': 1) 5000 rows, 12 columns
         2) No empty rows
         3) Zero duplicate observations
         4) Uniform distribution of qualitative and quantitative attributes
==> Data is extremely clean and can not be generalized to population, but attempt ML analysis anyways
==> drop 'User_ID', turn the 'Minutes Streamed Per Day' and 'Discover Weekly Engagement (%)' into two categories
==> segmentation based on most played artist
'''
data = pd.read_csv("Global_Music_Streaming_Listener_Preferences.csv")

df_nulls = pd.DataFrame({'col': data.columns,
                        'dtype': data.dtypes.values, #.values will match the col name to the dtypes and concat.
                        'null': data.isnull().sum()
                        }) 

num_duplicates = data.duplicated().sum()

# 'Country' ==> Uniform
df_countries = pd.DataFrame(data['Country'].value_counts() / data.shape[0] * 100)

# 'Streaming Platform' ==> Uniform
df_plat = pd.DataFrame(data['Streaming Platform'].value_counts() / data.shape[0] * 100)

# 'Most Played Artist' ==> Uniform
df_artist = pd.DataFrame(data['Most Played Artist'].value_counts() / data.shape[0] * 100)

# 'Top Genre' ==> Uniform
df_genre = pd.DataFrame(data['Top Genre'].value_counts() / data.shape[0] * 100)

# All unique values
df_attr = pd.DataFrame({'Variable': data.columns,
                        'Unique values': [len(data[x].unique()) for x in data.columns]
})
# print(df_attr)

activity = []
for i in data['Minutes Streamed Per Day']:
    if i < mean(data['Minutes Streamed Per Day']): # mean = 309.2372 minutes
        activity.append(0)
    else:
        activity.append(1)
data['Active Listener'] = activity

discoverability = []
for i in data['Discover Weekly Engagement (%)']:
    if i < 50: # mean ~ 50%
        discoverability.append(0)
    else:
        discoverability.append(1)
data['Discoverability'] = discoverability

#============================== PRE-PROCESSING ==============================

X = data.drop(['User_ID', 'Minutes Streamed Per Day', 'Discover Weekly Engagement (%)'], axis = 1)
y = data.loc[:,['Most Played Artist']]

# print(df_attr)
cat_col = ['Country', 'Streaming Platform', 'Top Genre', 'Active Listener', 'Subscription Type', 
           'Listening Time (Morning/Afternoon/Night)', 'Discoverability', 'Most Played Artist']
num_col = ['Age', 'Number of Songs Liked', 'Repeat Song Rate (%)']