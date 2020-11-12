import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

columns = ['user_id', 'artist_id','artist']
df = pd.read_csv('user_artists.csv', names=columns)
df.head()

artist_data = df[['user_id', 'artist']]

onehot = artist_data.pivot_table(index='user_id', columns= 'artist',aggfunc=len,fill_value=0)
onehot = onehot>0

plt.rcParams['figure.figsize'] = (10,6)
color = plt.cm.inferno(np.linspace(0,1,20))
df[['artist']].value_counts().head(20).plot.bar(color = color)
plt.title('Top 20 Most Frequent Items(Artists)')
plt.ylabel('Counts')
plt.xlabel('Artists')
plt.show()

import squarify
plt.rcParams['figure.figsize']=(10,10)
Items = df[['artist']].value_counts().head(20).to_frame()
size = Items[0].values
lab = Items.index
color = plt.cm.copper(np.linspace(0,1,20))
squarify.plot(sizes=size, label=lab, alpha = 0.7, color=color)
plt.title('Tree map of Most Frequent Items')
plt.axis('off')
plt.show()

from mlxtend.frequent_patterns import association_rules, apriori

# compute frequent items using the Apriori algorithm
frequent_itemsets = apriori(onehot, min_support = 0.1, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

# compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift",min_threshold=1)
rules.head()