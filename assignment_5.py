from sklearn import datasets
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
data = datasets.fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=['MedHouseVal'])


# Splitting test and training set, from https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
msk = np.random.rand(len(df)) <=0.8
X_train = df[msk]
y_train = target[msk]
X_test = df[~msk]
y_test = target[~msk]

num_trees = 250

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def get_random(X_train,y_train, X_test, y_test,m):
    mae = []
    tree = []
    for i in range(1, num_trees):
        print(i)
        tree.append(i)
        regressor = RandomForestRegressor(n_estimators=i, criterion = 'mae', max_features=m)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        each_mae = metrics.mean_absolute_error(y_test, y_pred)
        mae.append(each_mae)
    return mae

mae_2 = get_random(X_train,y_train, X_test, y_test,m=2)
mae_6 = get_random(X_train,y_train, X_test, y_test,m=6)

print("mae_2:", mae_2)
print("mae_6:", mae_6)
