import pandas as pd
import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt


class Data(object):
    def __init__(self):
        url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
        self.df = pd.read_csv(url, sep="\t")
        self.df.drop(self.df.columns[0], axis=1, inplace=True)
        self.df.drop(labels=["train"], axis=1, inplace=True)

#        self.data_validation = data[int(0.8*len(data)):int(0.9*len(data))]
#        self.data_test = data[int(0.9*len(data)):]

    @staticmethod
    def scaler(input_df):
        return input_df.subtract(input_df.mean()).divide(input_df.std(ddof=0))

    def get_data_train(self, add_prefix):
        length = int(0.8*len(self.df))
        data_training = Data.scaler(self.df[:length].drop(labels=["lpsa"], axis=1))
        data_target = self.df["lpsa"][:length]  # self.df["train"][:length].replace(["T", "F"], [1, 0])
        prefix = pd.DataFrame(data=np.full((length, 1), 1))
        if add_prefix:
            data_training.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_training), np.array(data_target)

    def get_data_test(self, add_prefix):
        length_min = int(0.8 * len(self.df))
        length_max = int(0.9 * len(self.df))
        data_testing = Data.scaler(self.df[length_min:length_max].drop(labels=["lpsa"], axis=1))
        data_target = self.df["lpsa"][length_min:length_max]
        prefix = np.ones((data_testing.shape[0], 1))
        if add_prefix:
            data_testing.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_testing), np.array(data_target)

    def get_data_validation(self, add_prefix):
        length_min = int(0.9 * len(self.df))
        data_validation = Data.scaler(self.df[length_min:].drop(labels=["lpsa"], axis=1))
        data_target = self.df["lpsa"][length_min:]
        prefix = np.ones((data_validation.shape[0], 1))
        if add_prefix:
            data_validation.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_validation), np.array(data_target)

    def get_correlation(self, label_a, label_b):
        length = len(self.df)
        x = np.array(self.df[label_a][:length])
        y = np.array(self.df[label_b][:length])
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        result = np.dot(x-x_mean, y-y_mean)/length / math.sqrt(np.var(x) * np.var(y))
        return result


def linear_regression(data):
    x_training, y_training = data.get_data_train(True)
    V = inv(np.dot(np.transpose(x_training), x_training))
    beta_hat = np.dot(np.dot(V, np.transpose(x_training)), y_training)
    y_hat = np.dot(x_training, beta_hat)
    N = len(x_training)
    p = len(x_training[0]) - 1
    variance_hat = np.sqrt(1 / (N - p - 1) * sum([(y_training[i] - y_hat[i])**2 for i in range(N)]))
    standard_err = variance_hat * np.sqrt(np.array([V[i][i] for i in range(len(V))]))
    Z = [beta_hat[j]/standard_err[j] for j in range(len(beta_hat))]
    x_header = ["Term", "Coefficient", "Std. Error", "Z Score"]
    y_header = ["Intercept", "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    for x in x_header:
        print("{:<15}".format(x), end="")
    print()
    for i in range(len(y_header)):
        print("{:<15}".format((y_header[i])), end="")
        print("{:<15}".format("{:.2f}".format(beta_hat[i])), end="")
        print("{:<15}".format("{:.2f}".format(standard_err[i])), end="")
        print("{:<15}".format("{:.2f}".format(Z[i])))
    print("")
    x_testing, y_testing = data.get_data_test(True)
    prediction = np.dot(x_testing, beta_hat)
    MSE = np.sum(np.power((prediction - y_testing), 2)) / len(prediction)
    print("Mean Square Error: {:.2f}".format(MSE.item()))


def ridge_regression(data):
    print("***********************PART 2***************************")
    x_training, y_training = data.get_data_train(False)
    x_testing, y_testing = data.get_data_test(False)
    x_validation, y_validation = data.get_data_validation(False)
    I = np.identity(len(x_training[0]))
    lambda_list = np.linspace(80, 100, 20)
    beta_list = []
    MSE_min = float('inf')
    lambda_opm = 0
    beta_opm = np.ones(len(x_training[0]))
    for lamb in lambda_list:
        beta_ridge = np.dot(np.dot(inv(np.add(np.dot(np.transpose(x_training), x_training), lamb*I).astype(np.int)), np.transpose(x_training)), y_training)
        prediction = np.dot(x_validation, beta_ridge)
        MSE = np.sum(np.power((prediction - y_validation), 2)) / len(prediction)
        if MSE < MSE_min:
            MSE_min = MSE
            lambda_opm = lamb
            beta_opm = beta_ridge
        beta_list.append(beta_ridge)
    beta_list = np.array(beta_list)
    print("Find Optimal Lambda as {:.2f}".format(lambda_opm) + "Find minimum Mean Square Error as {:.2f}".format(MSE_min))
    prediction = np.dot(x_testing, beta_opm)
    MSE = np.sum(np.power((prediction - y_testing), 2)) / len(prediction)
    print("Mean Square Error on testing dataset is {:.2f}".format(MSE))
    plt.figure(figsize=(10, 10))
    for i in range(beta_list.shape[1]):
        plt.plot(lambda_list.flatten(), beta_list[:,i].flatten(), "--o")
    plt.show()


def generate_correlation_table(x_header, y_header, data):
    print("***********************PART 1***************************")
    print("{:<15}".format(""), end="")
    for x in x_header:
        print("{:<15}".format(x), end="")
    print()
    for i in range(len(y_header)):
        print("{:<15}".format(y_header[i]), end="")
        for j in range(i+1):
            corr = data.get_correlation(y_header[i], x_header[j])
            print("{:<15}".format("{:.2f}".format(corr)), end="")
        print("")


def main():
    data = Data()
    x_header = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason"]
    y_header = ["lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    generate_correlation_table(x_header, y_header, data)
    print("\n")
    linear_regression(data)
    ridge_regression(data)


if __name__ == "__main__":
    main()
