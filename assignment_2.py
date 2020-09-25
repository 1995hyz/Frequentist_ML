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
        # data_target = Data.scaler(self.df[target_label][:length])
        data_target = self.df[target_label][:length]
        data_training.insert(loc=data_training.shape[1], column=target_label, value=data_target)
        prefix = pd.DataFrame(data=np.full((length, 1), 1))
        if add_prefix:
            data_training.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_training)

    def get_data_test(self, add_prefix, target_label):
        length_min = int(0.8 * len(self.df))
        length_max = int(0.9 * len(self.df))
        data_testing = Data.scaler(self.df[length_min:length_max].drop(labels=[target_label], axis=1))
        # data_target = Data.scaler(self.df[target_label][length_min:length_max])
        data_target = self.df[target_label][length_min:length_max]
        data_testing.insert(loc=data_testing.shape[1], column=target_label, value=data_target)
        prefix = np.ones((data_testing.shape[0], 1))
        if add_prefix:
            data_testing.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_testing)

    def get_data_validation(self, add_prefix, target_label):
        length_min = int(0.9 * len(self.df))
        data_validation = Data.scaler(self.df[length_min:].drop(labels=[target_label], axis=1))
        # data_target = Data.scaler(self.df[target_label][length_min:])
        data_target = self.df[target_label][length_min:]
        data_validation.insert(loc=data_validation.shape[1], column=target_label, value=data_target)
        prefix = np.ones((data_validation.shape[0], 1))
        if add_prefix:
            data_validation.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_validation)

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


def sgd_training(data, learning_rate, n_epoch, data_val):
    theta = [0.0] * (data.shape[1] - 1)
    error_min = float('inf')
    theta_optimal = np.copy(theta)
    feature_val = data_val[:, 0:-1]
    label_val = data_val[:, -1]
    for n in range(n_epoch):
        np.random.shuffle(data)
        feature = data[:, 0:-1]
        label = data[:, -1]
        error_sum = 0
        for i in range(feature.shape[0]):
            prediction = label_prediction(theta, feature[i])
            err = label[i] - prediction
            error_sum += err**2
            theta[0] = sgd_basic(1, err, learning_rate, theta[0])
            for j in range(feature.shape[1]-1):
                theta[j+1] = sgd_basic(feature[i][j], err, learning_rate, theta[j+1])
        # error_sum = get_validation_error(theta, feature_val, label_val)
        # error_sum = accuracy_test(theta, feature_val, label_val)
        if error_sum < error_min:
            error_min = error_sum
            theta_optimal = np.copy(theta)
        print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (n+1, learning_rate, error_sum))
    print(accuracy_test(theta_optimal, feature_val, label_val))
    return theta_optimal


def sgd_validation_L2(data, learning_rate, n_epoch):
    N = 100
    lamb = np.logspace(-20, 1, N)
    theta = np.zeros((N, data.shape[1] - 1))
    error_sum = np.zeros((N, 1))

    for n in range(n_epoch):
        np.random.shuffle(data)
        feature = data[:, 0:-1]
        label = data[:, -1]
        for i in range(feature.shape[0]):
          for l in range(N):
            prediction = label_prediction(theta[l, :], feature[i])
            err = label[i] - prediction
            error_sum[l] += err**2
            theta[l, 0] = sgd_basic(1, err, learning_rate, theta[l, 0])
            for j in range(feature.shape[1]-1):
                theta[l, j+1] = sgd_basic(feature[i][j], err, learning_rate, theta[l, j+1])
                theta[l, j+1] -= 2*lamb[l]*theta[l, j+1]
    print("Errors in Validation: ", error_sum)
    min = np.argmin(error_sum)
    optimalLamb = lamb[min]
    print("optimal Lambda: ", optimalLamb)
    return optimalLamb


def sgd_training_L2(data, learning_rate, n_epoch, lamb):
    theta = [0.0] * (data.shape[1] - 1)
    for n in range(n_epoch):
        np.random.shuffle(data)
        feature = data[:, 0:-1]
        label = data[:, -1]
        error_sum = 0
        for i in range(feature.shape[0]):
            prediction = label_prediction(theta, feature[i])
            err = label[i] - prediction
            error_sum += err**2
            theta[0] = sgd_basic(1, err, learning_rate, theta[0])
            for j in range(feature.shape[1]-1):
                theta[j+1] = sgd_basic(feature[i][j], err, learning_rate, theta[j+1])
                theta[j+1] -= 2*lamb*theta[j+1]
        print('>Training - epoch=%d, learning_rate=%.3f, error=%.3f' % (n, learning_rate, error_sum))
    return theta


def logistic_regression(data):
    data_train = data.get_data_train(True, "chd")
    data_validation = data.get_data_validation(True, "chd")
    data_test = data.get_data_test(True, "chd")
    learning_rate = 0.1
    n_epoch = 500
    theta = sgd_training(data_train, learning_rate, n_epoch, data_validation)
    accuracy = accuracy_test(theta, data_test[:, 0:-1], data_test[:, -1])
    print("*****")
    print(accuracy)
    #thetaL2 = sgd_training_L2(x_train, y_train, learning_rate, n_epoch,
    #                          sgd_validation_L2(x_validation, y_validation, learning_rate, n_epoch))


def sigmoid(num):
    return int(round(1 / (1 + math.exp(-num))))


def accuracy_test(theta, x, y):
    prediction = [sigmoid(label_prediction(theta, row)) for row in x]
    num_correct = 0
    for i in range(len(prediction)):
        if prediction[i] == y[i]:
            num_correct += 1
    return num_correct / len(prediction)


def get_validation_error(theta, x, y):
    error_sum = 0
    for i in range(x.shape[0]):
        prediction = label_prediction(theta, x[i])
        err = y[i] - prediction
        error_sum += err ** 2
    return error_sum


# Sigmoid function: g(z) = 1 / 1 + e^-z
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Logistic regression with no regularization to calculate theta
def logistic_noreg(x_train, y_train, num_steps, learning_rate):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    r, c = np.shape(x_train)
    # Initialize theta to zeros
    theta = np.zeros(c)

    # Stochastic gradient descent over num_steps
    for step in range(num_steps):
        index_array = list(range(r))
        j = 0
        while j < r:
            # Randomly select a single observation of x_train
            random = int(np.random.uniform(0, index_array.__len__()))
            x_row = x_train[index_array[random]]

            # Use the sigmoid function to calculate h
            z = np.dot(x_row, np.transpose(theta))
            h = sigmoid(z)
            # Calculate gradient and update theta
            gradient = np.dot((y_train[index_array[random]] - h), x_row)
            theta += learning_rate * gradient

            # Ensure the same observation does not get selected again in this step
            del (index_array[random])
            j += 1

    return theta

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
    # main()
    num_steps = 500
    learning_rate = 0.05

    # Calulate theta using train data
    theta_noreg = logistic_noreg(x_train, y_train, num_steps, learning_rate)

    # Calculate predictions on test data
    z_test = np.dot(x_test, np.transpose(theta_noreg))
    pred_noreg = np.round(sigmoid(z_test))
    acc_noreg = (pred_noreg == y_test).sum().astype(np.float128) / len(pred_noreg)
    print('Accuracy with no regularization: {0}'.format(acc_noreg))
