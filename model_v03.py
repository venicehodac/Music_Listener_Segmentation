'''
Continuation of model_v02.py:
- Random Forest Classfier w/ 5 classes
- Added parameter processing (Multilabel binarization)
- Extended hyperparameter tuning
'''

# load data
import pandas as pd
import numpy as np
# eda
import matplotlib.pyplot as plt
import seaborn as sns
import re
# pre-processing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

# Explore and clean/engineer each column:
# 1. 'music_expl_method' ==> into a list of multiple discovery methods
discover = data['music_expl_method'].str.split(r',[,\s]*')
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
# ==> Keep Playlist, recommendations, Radio, Others; drop ['Social media', 'Friends', 'Search', ''] in Pre-processing

# 2. 'Age' ==> Ordinal Encoder
ages = ['6-12', '12-20', '20-35', '35-60', '60+']
sns.displot(data['Age'])

# 3. 'Gender' ==> LabelEncoder
sns.displot(data['Gender'])
# print(sum(data['Gender']=='Others')) = 15/507 ~ 3% ==> Drop 'Others' gender
data = data.drop(data[data['Gender']=='Others'].index, axis = 0).reset_index(drop=True)

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
data = data.drop(data[data['fav_music_genre'].isin(['Classical & melody, dance', 'Old songs', 'trending songs random'])].index, axis=0).reset_index(drop=True)
# print(data['fav_music_genre'].value_counts()) ==> 8 genres; One Hot?

# 8. music_time_slot ==> ordinal
time_of_day = ['Morning', 'Afternoon', 'Night']

# 9. music_Influencial_mood: One Hot w/ music_Influencial_mood: Relaxation and stress relief, Uplifting and motivational, Sadness or melancholy, Social gatherings or parties
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
reason = data['music_lis_frequency'].str.split(r',[,\s]*')
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
plt.figure(7, (5,5))
sns.histplot(data['music_recc_rating'], binwidth=2.5)

plt.figure(8, (5,5))
sns.histplot(data['music_recc_rating'], binwidth=1)
# plt.show()

#=================================PRE-PROCESSING======================================
# Everything above is identical to model_v02.py
'''
Pre-processing:
1. 'Discovery Method' - OneHotEncoder() w/ Playlist, recommendations, Radio, Others (drop remaining) ==> MLB
2. 'Age' ==> OrdinalEncoder(categories = [ages])
3. 'Gender' ==> LabelEncoder ==> One Hot
4. 'spotify_usage_period' ==> Ordinal Encoder, categories = [[usage_period]]
5. 'Device' ==> OneHot w/ Smartphone, Computer or laptop, Smart speakers or voice assistants, Wearable devices ==> MLB
6. 'spotify_subscription_plan'  ==> Label Encoder ==> One Hot
7. 'fav_music_genre' ==> One Hot Encoding w/ 8 genres
8. 'music_time_slot' ==> Ordinal Encoder w/ categories = [time_of_day]
9. 'Mood' ==> One Hot Encoding ==> MLB
10. 'Reason' ==> One-Hot (Note: drop '' when pre-processing) ==> MLB
11. 'music_recc_rating' ==> STANDARDIZE.
'''
hot_col = ['Gender', 'spotify_subscription_plan', 'fav_music_genre']
mlb_col = ['Discovery Method', 'Device', 'Mood', 'Reason']

# Encode each multilabel binarizer in 'mlb_col'
def encode_mlb(col):
    # Encode
    data_copy = data.copy()
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(data_copy[col])
    # Get distinct variables for 'col'
    objs = mlb.classes_
    # Ammend global 'data' 
    data_copy = data_copy.drop(col, axis=1)
    encoded_df = pd.DataFrame(encoded, columns=objs)
    data_copy = pd.concat([data_copy, encoded_df], axis=1)
    return data_copy

# Amend global 'data' for all columns to be encoded with MultiLabelBinarizer()
for i in mlb_col:
    data = encode_mlb(i)

# ==> Keep attributes of interest only
data = data.drop(['', 'Friends', 'Search', 'Social media', 'Social gatherings ', 'Night time', 'when cooking', 'Random ', 'Before bed '], axis=1)
df_cols_2 = pd.DataFrame({'col': data.columns,
                        'dtype': data.dtypes.values, #.values will match the col name to the dtypes and concat.
                        }) 

hot_pipe = Pipeline(
    steps = [
        ('ohe', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
    ]
)

pre_processor = ColumnTransformer(
    transformers = [
        ('one hot', hot_pipe, hot_col),
        ('order age', OrdinalEncoder(categories=[ages]), ['Age']),
        ('order usage', OrdinalEncoder(categories=[usage_period]), ['spotify_usage_period']),
        ('order time', OrdinalEncoder(categories = [time_of_day]), ['music_time_slot'])
    ],
    remainder='passthrough',
    n_jobs=-1
)

classifier_v1 = Pipeline(
    steps = [
        ('preprocesser', pre_processor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ]
)
# Display Pipeline:
# set_config(display="diagram")
# print(classifier)

X = data.drop('music_recc_rating', axis = 1)
y = data['music_recc_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

classifier_v1.fit(X_train, y_train)

# Evaluate pipeline performance
print("Random Forest Classifier (with Multilabel Binarizer):")
print("Performance score:", classifier_v1.score(X_test, y_test))

# ============================== GridSearchCV ==============================
param_grid = {
    'n_estimators': [10, 20, 100, 200],
    'max_depth': [5, 10, 15]
}

X_encoded = pre_processor.fit_transform(X)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_encoded, y, test_size=0.5)
# X_train_2 = pre_processor.transform(X_train)
# X_test_2 = pre_processor.transform(X_test)

gridsearch = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
gridsearch.fit(X_train_2, y_train_2)
# Print best combination:
print("Best parameters:", gridsearch.best_params_)
print("Best CV score:", gridsearch.best_score_)

best_classifier = gridsearch.best_estimator_
y_pred = best_classifier.predict(X_test_2)

# Feature importance shows noisy data with high dimensionality ==> Revisit preprocessing stage.
feature_importance = best_classifier.feature_importances_
feature_name = pre_processor.get_feature_names_out()
sorted_indices = np.argsort(feature_importance)[::-1]

plt.figure(9,(10,5))
plt.bar(range(len(feature_importance)), feature_importance[sorted_indices], align='center')
plt.xticks(range(len(feature_importance)), np.array(feature_name)[sorted_indices], rotation=90)
plt.xlabel("Feature importance")
plt.title('Random Forest Feature Importance for Spotify Recc Rating')
# plt.show()

# Use the best model found
y_pred = best_classifier.predict(X_test_2)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_2, y_pred)
print(f"Accuracy on test set with best estimator: {accuracy}")

# Instead of 'accuracy' ==> try f1 score
from sklearn.metrics import f1_score

# Calculate F1 score = 0.25; POOR
f1 = f1_score(y_test_2, y_pred, average='macro')
print(f"F1 Score: {f1}")

plt.show()