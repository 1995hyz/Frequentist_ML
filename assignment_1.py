import pandas as pd
import numpy as np
from numpy.linalg import inv
import math


class Data(object):
    def __init__(self):
        url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
        self.df = pd.read_csv(url, delimiter="\t")
        self.df.drop(self.df.columns[0], axis=1, inplace=True)

#        self.data_validation = data[int(0.8*len(data)):int(0.9*len(data))]
#        self.data_test = data[int(0.9*len(data)):]

    def get_data_train(self, add_prefix):
        data_header = self.df.head()
        for header in data_header:
            if header != "train":
                col_mean = np.mean(self.df[header])
                col_sdd = np.std(self.df[header])
                self.df[header] = self.df[header].subtract(col_mean).div(col_sdd)
        length = int(0.8*len(self.df))
        data_training = self.df[:length]
        data_target = self.df["train"][:length].replace(["T", "F"], [1, 0])
        prefix = pd.DataFrame(data=np.full((length, 1), 1))
        data_training_x = data_training.drop(labels=["train"], axis=1)
        if add_prefix:
            data_training.insert(loc=0, column="prefix", value=prefix)
        return np.array(data_training_x), np.array(data_target)

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
    variance_hat = 1 / (N - p - 1) * sum([(y_training[i] - y_hat[i])**2 for i in range(N)])
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


def ridge_regression(data):
    x_training, y_training = data.get_data_train(False)
    I = np.identity(len(x_training[0]))
    lambda_list = np.linspace(0, 8, 20)
    beta_list = []
    for lamb in lambda_list:
        beta_ridge = np.dot(np.dot(np.invert(np.add(np.dot(np.transpose(x_training), x_training), lamb*I).astype(np.int)), np.transpose(x_training)), y_training)
        beta_list.append(beta_ridge)
    print(beta_list)


def generate_correlation_table(x_header, y_header, data):
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
