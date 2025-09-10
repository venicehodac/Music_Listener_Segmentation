"""
Classification/Segmentation Model #1:
Scikit Logistic Regression w/ NEW data

Classification of spotify users and what is the best way to introduce music to each group?
Clustering groups of users, then classify new users to each and test.
"""

# load data
import pandas as pd
import numpy as np
# eda
import matplotlib.pyplot as plt
import seaborn as sns
import re
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

# Exploring parameters with many unique parameters

# 1. 'music_expl_method' ==> into a list of multiple discovery methods
discover = data['music_expl_method'].str.split(r'[,\s]+')
data = data.drop('music_expl_method', axis=1)
data['Discovery Method'] = discover
# Visualization:
unique_methods = {}
for row in discover:
    for m in row:
        if m in unique_methods:
            unique_methods[m] = unique_methods[m] + 1
        else:
            unique_methods[m] = 1

unique_methods = pd.Series(unique_methods)
plt.figure(1, (10,5))
plt.bar(unique_methods.index, unique_methods.values)
# ==> Keep Playlist, recommendations, Radio, Others; drop ['Social', 'media', 'Friends', 'Search'] in Pre-processing

# 2. 'Age' ==> Ordinal Encoder
ages = ['6-12', '12-20', '20-35', '35-60', '60+']
sns.displot(data['Age'])

# 3. 'Gender' ==> LabelEncoder
sns.displot(data['Gender'])
# print(sum(data['Gender']=='Others')) = 15/507 ~ 3% ==> Drop 'Others' gender
data = data.drop(data[data['Gender']=='Others'].index, axis = 0)

# 4. spotify_usage_period ==> Ordinal Encoder
usage_period = ['Less than 6 months', '6 months to 1 year', '1 year to 2 years', 'More than 2 years']

# 5. spotify_listening_device ==> OneHot w/ Smartphone, Computer or laptop, Smart speakers or voice assistants, Wearable devices
device = data['spotify_listening_device'].str.split(r',[\s]+')
data = data.drop('spotify_listening_device', axis=1)
data['Device'] = device
# Visual:
unique_devices = {}
for row in data['Device']:
    for d in row:
        if d in unique_devices:
            unique_devices[d] = unique_devices[d] + 1
        else:
            unique_devices[d] = 1

unique_devices = pd.Series(unique_devices)
plt.figure(4,(11,5))
plt.bar(unique_devices.index, unique_devices.values)

# 6. spotify_subscription_plan  ==> Label Encoder
data['spotify_subscription_plan'] = data['spotify_subscription_plan'].apply(lambda x: 'Free' if x == "Free (ad-supported)" else 'Premium')

# 7. fav_music_genre ==> Drop insignificant values - Classical & melody, dance, Old songs, trending songs random (4 rows total)
data = data.drop(data[data['fav_music_genre'].isin(['Classical & melody, dance', 'Old songs', 'trending songs random'])].index, axis=0)
# print(data['fav_music_genre'].value_counts()) ==> 8 genres; One Hot?

# 8. music_time_slot ==> ordinal
time_of_day = ['Morning', 'Afternoon', 'Night']

# 9. music_Influencial_mood: One Hot w/ music_Influencial_mood, Relaxation and stress relief, Uplifting and motivational, Sadness or melancholy, Social gatherings or parties
mood = data['music_Influencial_mood'].str.split(', ')
data = data.drop('music_Influencial_mood', axis=1)
data['Mood'] = mood
#Visual:
unique_reason = {}
for row in data['Mood']:
    for i in row:
        if i in unique_reason:
            unique_reason[i] = unique_reason[i] + 1
        else:
            unique_reason[i] = 1
unique_reason = pd.Series(unique_reason)
plt.figure(5,(11,5))
plt.bar(unique_reason.index, unique_reason.values)

# 10. music_lis_frequency ==> One-Hot (Note: drop '' when pre-processing)
reason = data['music_lis_frequency'].str.split(r',[\s]*')
data = data.drop('music_lis_frequency', axis=1)
data['Reason'] = reason
# Visual:
unique_reason_2 = {}
for row in reason:
    for i in row:
        if i in unique_reason_2:
            unique_reason_2[i] = unique_reason_2[i] + 1
        else:
            unique_reason_2[i] = 1
unique_reason_2 = pd.Series(unique_reason_2)
plt.figure(6,(11,8))
plt.xticks(rotation=45)
plt.xlabel('Reasons & Mood for listening', fontsize=9)
plt.bar(unique_reason_2.index, unique_reason_2.values)

# 11. music_recc_rating ==> No change.
# sns.displot(data['music_recc_rating'], kde = True)
# plt.show()

#=================================PRE-PROCESSING======================================
'''
Pre-processing:
LabelEncoder() for non-ordinal items
1. 'music_recc_rating' ==> 'Discovery Method' - OneHotEncoder() w/ Playlist, recommendations, Radio, Others (drop remaining)
2. 'Age' ==> OrdinalEncoder(categories = [ages])
3. 
'''
df_cols_v2 = pd.DataFrame({'col': data.columns,
                        'dtype': data.dtypes.values,
                        }) 
print(df_cols_v2)


'''
Paremeters test #1: music_expl_method, music_recc_rating, fav_music_genre, spotify_subscription_plan
------------------------------------
1. LabelEncoder() since all are categorical
2. sns.heatmap
3. take most significant and perform classification model
4. check the feature selection
5. finalize classification
'''