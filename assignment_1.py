import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import warnings
from sklearn import linear_model
from datetime import datetime
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures


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


def linear_regression(data, target_label, y_header):
    """Simple linear regression model."""
    x_training, y_training = data.get_data_train(True, target_label)
    V = inv(np.dot(np.transpose(x_training), x_training))
    # Equation to calculate beta.
    beta_hat = np.dot(np.dot(V, np.transpose(x_training)), y_training)
    y_hat = np.dot(x_training, beta_hat)
    N = len(x_training)
    p = len(x_training[0]) - 1
    variance_hat = np.sqrt(1 / (N - p - 1) * sum([(y_training[i] - y_hat[i])**2 for i in range(N)]))
    standard_err = variance_hat * np.sqrt(np.array([V[i][i] for i in range(len(V))]))
    Z = [beta_hat[j]/standard_err[j] for j in range(len(beta_hat))]
    x_header = ["Term", "Coefficient", "Std. Error", "Z Score"]
    for x in x_header:
        print("{:<15}".format(x), end="")
    print()
    for i in range(len(y_header)):
        print("{:<15}".format((y_header[i])), end="")
        print("{:<15}".format("{:.2f}".format(beta_hat[i])), end="")
        print("{:<15}".format("{:.2f}".format(standard_err[i])), end="")
        print("{:<15}".format("{:.2f}".format(Z[i])))
    print("")
    x_testing, y_testing = data.get_data_test(True, target_label)
    prediction = np.dot(x_testing, beta_hat)
    MSE = np.sum(np.power((prediction - y_testing), 2)) / len(prediction)
    print("Mean Square Error on testing set: {:.3f}".format(MSE.item()))
    print("Mean Square Error on baseline: {:.3f}".format(get_baseline(y_training, y_testing)))


def ridge_regression(data, target_label):
    print("***********************PART 2***************************")
    x_training, y_training = data.get_data_train(False, target_label)
    x_testing, y_testing = data.get_data_test(False, target_label)
    x_validation, y_validation = data.get_data_validation(False, target_label)
    I = np.identity(len(x_training[0]))
    lambda_list = np.linspace(0, 1, 100)
    beta_list = []
    MSE_min = float('inf')
    lambda_opm = 0
    beta_opm = np.ones(len(x_training[0]))
    for lamb in lambda_list:
        # Equation to calculate beta.
        beta_ridge = np.dot(np.dot(inv(np.add(np.dot(np.transpose(x_training), x_training), lamb*I)), np.transpose(x_training)), y_training)
        prediction = np.dot(x_validation, beta_ridge)
        MSE = np.sum(np.power((prediction - y_validation), 2)) / len(prediction)
        if MSE < MSE_min:
            MSE_min = MSE
            lambda_opm = lamb
            beta_opm = beta_ridge
        beta_list.append(beta_ridge)
    beta_list = np.array(beta_list)
    print("Find Optimal Lambda as {:.3f}".format(lambda_opm) + " Find minimum Mean Square Error as {:.3f}".format(MSE_min))
    prediction = np.dot(x_testing, beta_opm)
    MSE = np.sum(np.power((prediction - y_testing), 2)) / len(prediction)
    print("Mean Square Error on testing dataset is {:.3f}".format(MSE))
    plt.figure(figsize=(10, 10))
    for i in range(beta_list.shape[1]):
        plt.plot(lambda_list.flatten(), beta_list[:, i].flatten(), "--o")
    plt.legend(data.column_name.drop(labels=[target_label]))
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.title("Ridge Regression")
    plt.show()


def lasso_regression(data, target_label):
    print("***********************PART 3***************************")
    x_training, y_training = data.get_data_train(False, target_label)
    x_testing, y_testing = data.get_data_test(False, target_label)
    x_validation, y_validation = data.get_data_validation(False, target_label)
    lambda_list = np.linspace(0, 0.05, 100)
    beta_list = []
    MSE_min = float('inf')
    lambda_opm = 0
    beta_opm = np.ones(len(x_training[0]))
    for lamb in lambda_list:
        model = linear_model.Lasso(alpha=lamb)
        model.fit(x_training, y_training)
        prediction = model.predict(x_validation)
        MSE = np.sum(np.power((prediction - y_validation), 2)) / len(prediction)
        if MSE < MSE_min:
            MSE_min = MSE
            lambda_opm = lamb
            beta_opm = model.coef_
        beta_list.append(model.coef_)
    beta_list = np.array(beta_list)
    print("Find Optimal Lambda as {:.3f}".format(lambda_opm) + " Find minimum Mean Square Error as {:.3f}".format(MSE_min))
    plt.figure(figsize=(10, 10))
    for i in range(beta_list.shape[1]):
        plt.plot(lambda_list.flatten(), beta_list[:, i].flatten(), "--o")
    plt.legend(data.column_name.drop(labels=[target_label]))
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.title("Lasso Regression")
    plt.show()
    prediction = np.dot(x_testing, beta_opm)
    MSE = np.sum(np.power((prediction - y_testing), 2)) / len(prediction)
    print("Mean Square Error on testing dataset is {:.3f}".format(MSE))


def lasso_regression_non_linear(data, target_label):
    print("***********************Stretch Goal***************************")
    x_training, y_training = data.get_data_train(False, target_label)
    x_testing, y_testing = data.get_data_test(False, target_label)
    x_validation, y_validation = data.get_data_validation(False, target_label)
    lambda_list = np.linspace(0, 0.05, 100)
    beta_list = []
    MSE_min = float('inf')
    lambda_opm = 0
    beta_opm = np.ones(len(x_training[0]))
    for lamb in lambda_list:
        model = linear_model.Lasso(alpha=lamb)
        model.fit(x_training, y_training)
        prediction = model.predict(x_validation)
        MSE = np.sum(np.power((prediction - y_validation), 2)) / len(prediction)
        if MSE < MSE_min:
            MSE_min = MSE
            lambda_opm = lamb
            beta_opm = model.coef_
        beta_list.append(model.coef_)
    print("Find Optimal Lambda as {:.3f}".format(lambda_opm) + " Find minimum Mean Square Error as {:.3f}".format(MSE_min))
    prediction = np.dot(x_testing, beta_opm)
    MSE = np.sum(np.power((prediction - y_testing), 2)) / len(prediction)
    print("Mean Square Error on testing dataset is {:.3f}".format(MSE))


def generate_correlation_table(x_header, y_header, data):
    """Display a correlation table of data."""
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


def add_feature_interaction(df, label):
    target = np.array(df[label])
    features = df.drop(labels=[label], axis=1)
    interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction = interaction.fit_transform(features)
    features_column_name = list(features.columns) + [str(i) for i in range(interaction.shape[1]-features.shape[1])]
    features_row_name = [str(i) for i in range(interaction.shape[0])]
    interaction = pd.DataFrame(data=interaction, index=features_row_name, columns=features_column_name)
    interaction.insert(loc=(interaction.shape[1]-1), column=label, value=target)
    return interaction


def square_data_input(df, label):
    target = np.array(df[label])
    features = df.drop(labels=[label], axis=1)
    features = features.pow(2)
    features.insert(loc=(features.shape[1]-1), column=label, value=target)
    return features


def load_dataset_1():
    url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
    df = pd.read_csv(url, sep="\t")
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(labels=["train"], axis=1, inplace=True)
    return df


def load_dataset_2():
    url = 'https://raw.githubusercontent.com/1995hyz/Frequentist_ML/master/Pima%20Indians%20Diabetes%20Database.csv'
    df = pd.read_csv(url, sep=",")
    return df


def main():
    random.seed(datetime.now())
    df = load_dataset_1()
    # Uncomment this line to randomize dataset.
    df = df.sample(frac=1, random_state=random.randint(0, 200)).reset_index().drop(labels=["index"], axis=1)
    data = Data(df)
    x_header = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason"]
    y_header = ["lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    generate_correlation_table(x_header, y_header, data)
    print("\n")
    y_header_regression = ["Intercept", "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    linear_regression(data, "lpsa", y_header_regression)
    ridge_regression(data, "lpsa")
    lasso_regression(data, "lpsa")
    data_2 = Data(load_dataset_2())
    x_header_2 = ["NTP", "PGC", "DBP", "SI", "BMI", "DPF"]
    y_header_2 = ["PGC", "DBP", "SI", "BMI", "DPF", "Age"]
    y_header_regression_2 = ["Intercept", "NTP", "PGC", "DBP", "SI", "BMI", "DPF", "Age"]
    generate_correlation_table(x_header_2, y_header_2, data_2)
    print("\n")
    linear_regression(data_2, "Class", y_header_regression_2)
    ridge_regression(data_2, "Class")
    lasso_regression(data_2, "Class")
    data_feature_squared = Data(square_data_input(df, "lpsa"))
    data_feature_added = Data(add_feature_interaction(df, "lpsa"))
    lasso_regression_non_linear(data_feature_squared, "lpsa")
    lasso_regression_non_linear(data_feature_added, "lpsa")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
