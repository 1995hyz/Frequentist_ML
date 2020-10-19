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


def load_dataset_2():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None, names=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ])
    df['species'] = df['species'].astype('category').cat.codes
    return df


def label_prediction(theta, x):
    h = 1 / (1 + np.exp(-np.dot(np.transpose(theta), x)))
    return h


def sgd_basic(feature, err, learning_rate, weight):
    """Update weight using basic stochastic gradient descent."""
    weight = weight + learning_rate * err * feature
    return weight


def sgd_training(data, learning_rate, n_epoch):
    theta = [0.0] * (data.shape[1] - 1)
    feature = data[:, 0:-1]
    label = data[:, -1]
    for n in range(n_epoch):
        for j in range(feature.shape[1]):
            index = random.randint(0, feature.shape[0]-1)
            prediction = label_prediction(theta, feature[index])
            err = label[index] - int(round(sigmoid(prediction)))
            theta = sgd_basic(feature[index], err, learning_rate, theta)
    return theta


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
    learning_rate = 0.06
    n_epoch = 1500
    get_baseline(data_train, data_test)
    # theta = sgd_training(data_train, learning_rate, n_epoch)
    # accuracy = accuracy_test(theta, data_test[:, 0:-1], data_test[:, -1])
    # print("Test Accuracy with unregularized sgd: ")
    # print(accuracy)
    # thetaL2 = sgd_training_L2(data_train, learning_rate, n_epoch,
    #                             sgd_validation_L2(data_validation, learning_rate, n_epoch))
    data_train = data.get_data_train(False, "chd")
    data_validation = data.get_data_validation(False, "chd")
    feature_header = list(data.column_name)
    feature_header.remove("chd")
    feature_header.append("chd")    # To make sure the label is at the end of the header list
    theta, feature_selected = sgd_stepwise(data_train, feature_header, learning_rate, n_epoch, data_validation)
    print(feature_selected)
    feature_unselected = pd.DataFrame(data=data_test[:, 0:-1], columns=(["prefix"]+feature_header[0:-1]))
    print(theta)
    accuracy = accuracy_test(theta, np.array(feature_unselected[feature_selected]), data_test[:, -1])
    print(accuracy)


def sigmoid(num):
    return 1 / (1 + math.exp(-num))


def accuracy_test(theta, x, y):
    prediction = [int(round(sigmoid(label_prediction(theta, row)))) for row in x]
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


def sgd_stepwise(data, data_header, learning_rate, n_epoch, data_val):
    feature = data[:, 0:-1]
    label = data[:, -1]
    feature_unselected = pd.DataFrame(data=feature, columns=data_header[0:-1])
    feature_unselected_header = list(feature_unselected.columns)
    feature_selected = pd.DataFrame(data=np.array([[1.0] * data.shape[0]]).T, columns=["prefix"])
    data_val_frame = pd.DataFrame(data=data_val, columns=data_header)
    feature_selected_val = pd.DataFrame(data=np.array(np.array([[1.0] * data_val.shape[0]]).T), columns=["prefix"])
    theta_opt = []
    accuracy_opt = 0
    while True:
        accuracy_opt_per_feature = 0
        theta_opt_per_feature = []
        feature_selected_candidate = feature_unselected_header[0]
        for header in feature_unselected_header:
            data_train = feature_selected.copy()
            data_train.insert(len(data_train.columns), header, feature_unselected[header], True)
            data_train.insert(len(data_train.columns), "label", label, True)
            theta = sgd_training(np.array(data_train), learning_rate, n_epoch)
            data_validation = feature_selected_val.copy()
            data_validation.insert(len(data_validation.columns), header, data_val_frame[header], True)
            data_validation.insert(len(data_validation.columns), "label", data_val[:, -1], True)
            accuracy = accuracy_test(theta, np.array(data_validation)[:, 0:-1], np.array(data_validation)[:, -1])
            if accuracy > accuracy_opt_per_feature:
                accuracy_opt_per_feature = accuracy
                feature_selected_candidate = header
                theta_opt_per_feature = theta
        if accuracy_opt_per_feature > accuracy_opt:
            accuracy_opt = accuracy_opt_per_feature
            theta_opt = theta_opt_per_feature
            feature_selected.insert(len(feature_selected.columns), feature_selected_candidate, feature_unselected[feature_selected_candidate], True)
            feature_selected_val.insert(len(feature_selected_val.columns), feature_selected_candidate, data_val_frame[feature_selected_candidate], True)
            del feature_unselected[feature_selected_candidate]
            feature_unselected_header.remove(feature_selected_candidate)
        else:
            break
    return theta_opt, feature_selected.columns


def sgd_multivariable(data, learning_rate, n_epoch):
    theta_list = []
    feature = data[:, 0:-1]
    label = data[:, -1]
    classes = np.unique(label)
    for single_class in classes:
        label_training = np.array([np.where(label == single_class, 1, 0)]).T
        data_training = np.append(feature, label_training, axis=1)
        theta = sgd_training(data_training, learning_rate, n_epoch)
        theta_list.append(theta)
    return theta_list


def label_prediction_multivariable(data, thetas):
    feature = data[:, 0:-1]
    label = data[:, -1]
    prediction_list = []
    for i in range(feature.shape[0]):
      prediction = np.argmax([sigmoid(np.dot(feature[i], np.transpose(theta))) for theta in thetas])
      prediction_list.append(prediction)
    return np.array(prediction_list == label).mean()


def main():
    random.seed(datetime.now())
    df = load_dataset_1()
    # scatter_plot = seaborn.pairplot(df, hue = "chd")
    # scatter_plot.fig. suptitle("Scatterplot Matrix of the South African Heart Disease Data", y = 1)
    # plt.show()
    # Uncomment this line to randomize dataset.
    df = df.sample(frac=1, random_state=random.randint(0, 200)).reset_index().drop(labels=["index"], axis=1)
    data = Data(df)
    logistic_regression(data)

    # df2 = load_dataset_2()
    # df2 = df2.sample(frac=1, random_state=random.randint(0, 200)).reset_index()
    # data2 = Data(df2)
    # data_train = data2.get_data_train(True, "species")
    # data_test = data2.get_data_test(True, "species")
    # theta_list = sgd_multivariable(data_train, learning_rate=0.06, n_epoch=1500)
    # accuracy = label_prediction_multivariable(data_test, theta_list)


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()


def load_dataset_3():
    url = "https://raw.githubusercontent.com/1995hyz/Frequentist_ML/master/column_2C_weka.csv"
    df = pd.read_csv(url, sep=",")
    df['class'] = df['class'].astype('category').cat.codes
    return df

df = load_dataset_3()
print("")

# Uncomment this line to randomize dataset.
df = df.sample(frac=1, random_state=random.randint(0, 200)).reset_index().drop(labels=["index"], axis=1)
data = Data(df)
data_train = data.get_data_train(True, "class")
data_validation = data.get_data_validation(True, "class")
data_test = data.get_data_test(True, "class")
learning_rate = 0.05
n_epoch = 1500
baselineAccuracy = get_baseline(data_train, data_test)
theta = sgd_training(data_train, learning_rate, n_epoch)
accuracy = accuracy_test(theta, data_test[:, 0:-1], data_test[:, -1])
print("Test Accuracy with unregularized sgd: ", accuracy)
accuracyL2, thetaL2 = sgd_validation_L2(data_train, data_validation,learning_rate, n_epoch)
accuracyL1, lasso = logistic_regression_L1(data_train, data_validation, learning_rate, n_epoch)
accuracyStep = logistic_regression_foward_step(data)
print("Forward-step regression: ", accuracyStep)
approach = ['Baseline', 'Unregularized', 'with L2 penalty', 'with L1 pentalty', 'Forward Stepwise']
correctness = [baselineAccuracy, accuracy, accuracyL2, accuracyL1,accuracyStep]
tableCorrectness = pd.DataFrame(correctness,approach)
print('% correctness for different approach of logistic regression')
display(tableCorrectness)
#Create a table for all the % of correctness