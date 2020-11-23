"""columns = ['user_id', 'artist_id', 'artist', "plays"]
df = pd.read_csv('usersha1-artmbid-artname-plays-short.tsv', names=columns, sep='\t')
columns_profile = ['user_id', 'gender', 'age', 'country', 'signup']
df_profile = pd.read_csv('usersha1-profile.tsv', names=columns_profile, sep='\t')
df.head()

artist_data = df[['user_id', 'artist']]

output_header = ['user_id', 'artist_id', 'artist', 'gender', 'age', 'country', 'signup']
df_output = pd.DataFrame(columns=output_header)

for i in range(len(df)):
    row = df.iloc[i, :]
    row_profile = df_profile.loc[df_profile['user_id'] == row['user_id']].iloc[0]
    if row_profile is not None:
        df_output = df_output.append({'user_id': row['user_id'], 'artist_id': row['artist_id'], 'artist': row['artist'],
                                      'gender': row_profile['gender'], 'age': row_profile['age'],
                                      'country': row_profile['country'], 'signup': row_profile['signup']},
                                     ignore_index=True)

df_output.to_csv("user_song_and_profile.csv")"""

"""columns = ['user_id', 'artist', 'gender', 'country']
df = pd.read_csv('user_song_and_profile_usa.csv', names=columns)
df = df[df.gender.notnull()]
df.head()

df_output = pd.DataFrame(columns=columns)

for i in range(len(df)):
    row = df.iloc[i, :]
    if row['country'] == "United States":
        df_output = df_output.append({'user_id': row['user_id'], 'artist': row['artist'], 'gender': row['gender'],
                                      'country': row['country']}, ignore_index=True)
df_output.to_csv("user_song_and_profile_usa_processed.csv")"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

columns = ['user_id', 'artist', 'gender']
df = pd.read_csv('user_song_and_profile_usa_processed.csv', names=columns)
df = df[df.gender.notnull()]

artist_data = df[['user_id', 'artist']]
gender_data = df[['user_id', 'gender']]
onehot = artist_data.pivot_table(index='user_id', columns='artist', aggfunc=len, fill_value=0)
onehot = onehot > 0
onehot_f = onehot.copy()

gender_data = gender_data.drop_duplicates()
female_data = gender_data.copy()

gender_data['gender'] = gender_data['gender'].replace('f', False)
gender_data['gender'] = gender_data['gender'].replace('m', True)
gender = gender_data['gender'].tolist()

female_data['gender'] = female_data['gender'].replace('f', True)
female_data['gender'] = female_data['gender'].replace('m', False)
gender_f = female_data['gender'].tolist()
onehot_f['gender'] = gender_f

female_countering = 0

for gender_item in gender:
    if not gender_item:
        female_countering = female_countering + 1

print(female_countering)
print(len(gender) - female_countering)
