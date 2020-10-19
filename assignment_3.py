from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from random import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import statistics

N = 50
p = 5000

#predictor = np.zeros((N, p))
#labels = np.zeros((N, 1))
#
#n_simulation = 50
#cv_score_wrong = np.zeros((n_simulation, 5))
#for n in range(n_simulation):
#    for each in range(N):
#        labels[each] = randint(0, 1)
#        predictor[each] = np.random.normal(0, 1, p)
#
#    cor = np.zeros((p, 1))
#
#    for i in range(p):
#        x = predictor[:, i].reshape(50, 1)
#        cor[i] = np.corrcoef(x.T, labels.T)[1, 0]
#
#    sorted = np.sort(cor, axis=0)
#    sorted = sorted[4900:]
#    indexBest = np.zeros((100, 1))
#    for j in range(100):
#        for k in range(p):
#            if cor[k] == sorted[j]:
#                indexBest[j] = k
#                continue
#
#    indexBest.astype(np.int64)
#    bestPredictor = np.zeros((50, 100))
#    for i in range(100):
#        bestPredictor[:, i] = predictor[:, indexBest[i]]
#
#    incorrectNeighbor = KNeighborsClassifier(n_neighbors=1)
#    cv_score_wrong[n, :] = cross_val_score(incorrectNeighbor, bestPredictor, labels)
#
#print("Accuracy: ", sum(cv_score_wrong)/len(cv_score_wrong))

n_simulation = 50
score_correct = np.zeros((1, n_simulation))
for n in range(n_simulation):
    predictor_raw = np.zeros((N, p))
    predictor = list()
    labels = np.zeros((N, 1))
    scores = np.zeros((1, 5))
    for each in range(N):
        labels[each] = randint(0, 1)
        predictor_raw[each] = np.random.normal(0, 1, p)
    # Split training data into five parts
    for i in range(5):
        predictor.append(predictor_raw[i*10:(i+1)*10, :])
    predictor = np.array(predictor)
    # Loop through each part as validation set.
    for k in range(5):
        validation = predictor[k, :, :]
        validation_label = labels[k*10:(k+1)*10, :]
        training_indexes = list(range(5))
        training_indexes.remove(k)
        training_label = np.vstack((labels[:k*10, :], labels[(k+1)*10:, :]))
        training = np.vstack(tuple([predictor[j, :, :] for j in training_indexes]))
        cor = np.zeros((p, 1))

        for i in range(p):
            x = training[:, i].reshape(40, 1)
            cor[i] = np.corrcoef(x.T, training_label.T)[1, 0]

        sorted = np.sort(cor, axis=0)
        sorted = sorted[4900:]
        index_best = np.zeros((100, 1))
        for j in range(100):
            for b in range(p):
                if cor[b] == sorted[j]:
                    index_best[j] = b
        index = index_best.astype(int)
        best_predictor = np.zeros((40, 100))
        best_predictor_val = np.zeros((10, 100))
        for i in range(100):
            best_predictor[:, i] = training[:, index[i]].ravel()
            best_predictor_val[:, i] = validation[:, index[i]].ravel()
        # Use multi-variable logistic regression to do classification
        clf = LogisticRegression(random_state=0).fit(best_predictor, training_label)
        score = clf.score(best_predictor_val, validation_label)
        scores[0, k] = score
    score_correct[0, n] = np.mean(scores)
print("Accuracy: ", np.mean(score_correct))
