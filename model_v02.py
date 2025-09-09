"""
Classification/Segmentation Model #1:
Scikit Logistic Regression w/ NEW data

Classification of spotify users and what is the best way to introduce music to each group?
Clustering groups of users, then classify new users to each and test.
"""

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
from sklearn.linear_model import LogisticRegression

#============================== EDA for new data ==============================

data = pd.read_excel("data/Spotify_data.xlsx")
# 520 rows x 20 cols

df_nulls = pd.DataFrame({'col': data.columns,
                        'dtype': data.dtypes.values, #.values will match the col name to the dtypes and concat.
                        'null': data.isnull().sum()
                        }) 

data = data.drop(['preffered_premium_plan', 'pod_lis_frequency', 'fav_pod_genre', 'preffered_pod_format', 'pod_host_preference', 
                  'preffered_pod_duration', 'pod_variety_satisfaction', 'premium_sub_willingness', 'preferred_listening_content', ], axis=1)

df_cols = pd.DataFrame({'col': data.columns,
                        'dtype': data.dtypes.values, #.values will match the col name to the dtypes and concat.
                        }) 

# print(data.duplicated().sum()) ==> drop 13 duplicates
data = data.drop_duplicates()

df_attr = pd.DataFrame({'Variable': data.columns,
                        'Unique values': [len(data[x].unique()) for x in data.columns]
})

print(df_cols)
print(df_attr)
















'''
Paremeters test #1: music_expl_method, music_recc_rating, fav_music_genre, spotify_subscription_plan
------------------------------------
1. LabelEncoder() since all are categorical
2. sns.heatmap
3. take most significant and perform classification model
4. check the feature selection
5. finalize classification
'''

# param_1 = ['music_expl_method', 'music_recc_rating', 'fav_music_genre', 'spotify_subscription_plan']
# data_v1 = data.loc[:,['music_expl_method', 'music_recc_rating', 'fav_music_genre', 'spotify_subscription_plan']]

