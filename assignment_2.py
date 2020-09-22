import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import warnings
import seaborn
from sklearn import linear_model
from datetime import datetime
from numpy.linalg import inv


class Data(object):
    def __init__(self, data_frame):
        self.df = data_frame
        self.column_name = self.df.columns

    @staticmethod
    def scaler(input_df):
        """Normalize data by subtracting the mean and being divided by the variance."""
        # return input_df.subtract(input_df.mean()).divide(input_df.std(ddof=0))
        return (input_df-input_df.min())/(input_df.max()-input_df.min())

    def get_data_train(self, add_prefix, target_label):
        length = int(0.8*len(self.df))
        data_training = Data.scaler(self.df[:length].drop(labels=[target_label], axis=1))
        data_target = Data.scaler(self.df[target_label][:length])
        prefix = pd.DataFrame(data=np.full((length, 1), 1))
        if add_prefix:
            data_training.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_training), np.array(data_target)

    def get_data_test(self, add_prefix, target_label):
        length_min = int(0.8 * len(self.df))
        length_max = int(0.9 * len(self.df))
        data_testing = Data.scaler(self.df[length_min:length_max].drop(labels=[target_label], axis=1))
        data_target = Data.scaler(self.df[target_label][length_min:length_max])
        prefix = np.ones((data_testing.shape[0], 1))
        if add_prefix:
            data_testing.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_testing), np.array(data_target)

    def get_data_validation(self, add_prefix, target_label):
        length_min = int(0.9 * len(self.df))
        data_validation = Data.scaler(self.df[length_min:].drop(labels=[target_label], axis=1))
        data_target = Data.scaler(self.df[target_label][length_min:])
        prefix = np.ones((data_validation.shape[0], 1))
        if add_prefix:
            data_validation.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_validation), np.array(data_target)

    def get_correlation(self, label_a, label_b):
        """Given two labels, calculate the correlation between the two labelled columns."""
        length = len(self.df)
        x = np.array(self.df[label_a][:length])
        y = np.array(self.df[label_b][:length])
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        result = np.dot(x-x_mean, y-y_mean)/length / math.sqrt(np.var(x) * np.var(y))
        return result


def get_baseline(label_training, label_testing):
    """Calculate the baseline MSE by using the mean of label in training set as the prediction."""
    mean_training_label = np.mean(label_training)
    MSE = np.sum(np.power((mean_training_label - label_testing), 2)) / len(label_testing)
    return MSE


def load_dataset_1():
    url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data'
    df = pd.read_csv(url, sep=",")
    df.drop(df.columns[0], axis=1, inplace=True)
    mapping = {'Present': 1, 'Absent': 0}
    df = df.replace({'famhist': mapping})
    return df


def label_prediction(theta, x):
    h = 1 / (1 + np.exp(-np.dot(np.transpose(theta), x)))
    return h


def sgd_basic(feature, err, learning_rate, weight):
    """Update weight using basic stochastic gradient descent."""
    weight = weight + learning_rate * err * feature
    return weight


def sgd_training(feature, label, learning_rate, n_epoch):
    theta = [0.0] * feature.shape[1]
    error_min = float('inf')
    theta_optimal = theta
    for n in range(n_epoch):
        error_sum = 0
        for i in range(feature.shape[0]):
            prediction = label_prediction(theta, feature[i])
            err = label[i] - prediction
            error_sum += err**2
            theta[0] = sgd_basic(1, err, learning_rate, theta[0])
            for j in range(feature.shape[1]-1):
                theta[j+1] = sgd_basic(feature[i][j], err, learning_rate, theta[j+1])
        if error_sum < error_min:
            error_min = error_sum
            theta_optimal = theta
        print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (n, learning_rate, error_sum))
    print("*************")
    print(error_min)
    return theta_optimal


def logistic_regression(data):
    x_train, y_train = data.get_data_train(True, "chd")
    x_validation, y_validation = data.get_data_validation(True, "chd")
    x_test, y_test = data.get_data_test(True, "chd")
    learning_rate = 0.01
    n_epoch = 10000
    theta = sgd_training(x_train, y_train, learning_rate, n_epoch)
    accuracy = accuracy_test(theta, x_validation, y_validation)
    print(accuracy)
    print(accuracy_test(theta, x_test, y_test))


def sigmoid(num):
    return int(round(1 / (1 + math.exp(-num))))


def accuracy_test(theta, x, y):
    prediction = [sigmoid(label_prediction(theta, row)) for row in x]
    num_correct = 0
    for i in range(len(prediction)):
        if prediction[i] == y[i]:
            num_correct += 1
    return num_correct / len(prediction)


def main():
    random.seed(datetime.now())
    df = load_dataset_1()
    # seaborn.pairplot(df)
    # plt.show()
    # Uncomment this line to randomize dataset.
    # df = df.sample(frac=1, random_state=random.randint(0, 200)).reset_index().drop(labels=["index"], axis=1)
    data = Data(df)
    logistic_regression(data)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
