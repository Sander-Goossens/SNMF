import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import random
import os
from src.processing.bootstrapping import bootstrap_perSample
from src.processing.bootstrapping import bootstrap_equal

os.chdir("C:/Users/sande/PycharmProjects/MEP_data")


def data_split(N):
    data = pd.read_pickle("data/zou2021/zou_96SBS_filtered.pkl")
    features = data.columns[0:96]
    genes = data['Gene_KO'].unique().tolist()
    n_samples = data.size

    # Define class labels
    # MMR
    c1 = ['MLH1', 'MSH2', 'MSH6', 'PMS2']
    # HR
    c2 = ['EXO1', 'RNF168']
    # ... other classes

    # cc = Control Class
    cc = []
    for g in genes:
        if g not in (c1 + c2):
            cc.append(g)

    classes = [cc, c1, c2]

    data["label"] = -1
    for index, row in data.iterrows():
        for i in range(len(classes)):
            if row['Gene_KO'] in classes[i]:
                data.loc[index, "label"] = i
                break

    temp = data.loc[data['label'] == -1]

    test_idx = []

    # Select 1 sample per Class/Gene_KO for test / rest is train.
    # TODO: Train/Test -> 80/20% (with min 1 sample in test)
    for g in genes:
        # Number of samples in Gene_KO
        samples = data.loc[data["Gene_KO"] == g]
        test_idx.append(random.choice(samples.index))
        # select random sample

    test = data.loc[test_idx]
    train = data.drop(test_idx)

    train = bootstrap_perSample(train, "_", N)

    test = bootstrap_perSample(test, "_", N)

    X_train = train.loc[:, features].values
    y_train = train.loc[:, "label"].values

    X_test = test.loc[:, features].values
    y_test = test.loc[:, "label"].values

    return X_train, X_test, y_train, y_test


def run_MLR(X_train, X_test, y_train, y_test):
    '''

    :param N: number of Bootstrapped samples per real sample
    :return: data and model
    '''

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(X_train, y_train)

    return X_train, y_train, X_test, y_test, lr


def y_toND(y):
    print(len(y))
    print(max(y))
    new_y = np.zeros((len(y), max(y)))
    for i in range(len(y)):
        new_y[i, y[i]] = 1
    return new_y


def data_split_equal(N, nested):
    '''

    :param N:
    :param nested: Boolean
    :return:
    '''
    data = pd.read_pickle("data/zou2021/zou_96SBS_filtered.pkl")
    features = data.columns[0:96]
    genes = data['Gene_KO'].unique().tolist()
    n_samples = data.size

    # Control
    c0 = ['ATP2B4']
    # MMR
    c1 = ['MLH1', 'MSH2', 'MSH6', 'PMS2', 'PMS1']
    # HR
    c2 = ['EXO1', 'RNF168']
    # BER
    c3 = ['OGG1', 'UNG']

    classes = [c0, c1, c2, c3]

    data["label"] = -1
    for index, row in data.iterrows():
        for i in range(len(classes)):
            if row['Gene_KO'] in classes[i]:
                data.loc[index, "label"] = i
                break

    temp = data.loc[data['label'] == -1]

    test_idx = []

    # outer Test Set
    if nested:
        for g in genes:
            # Number of samples in Gene_KO
            samples = list(data.loc[data["Gene_KO"] == g].index)
            # for control cell line -> 2 test samples
            if g == 'ATP2B4':
                test_idx.extend(random.sample(samples, 2))
            else:
                test_idx.extend(random.sample(samples, 1))



    test = data.loc[test_idx]
    train = data.drop(test_idx)

    f_idx = [[] for i in range(3)]
    train_folds = [None] * 3

    for g in genes:
        # Split remaining Samples evenly over folds
        # If not divisable by 3 -> remaining 1 or 2 in random fold

        samples = train.loc[train["Gene_KO"] == g]
        samples_idx = list(samples.index)
        random.shuffle(samples_idx)
        n = len(samples_idx)

        min = math.floor(n / 3)
        mod = n % 3
        fold = [0, 1, 2]
        random.shuffle(fold)

        pos = 0
        for i in range(len(fold)):
            if mod > 0:
                new_pos = pos + min + 1
                f_idx[fold[i]].extend(samples_idx[ pos: new_pos ])
                mod -= 1
            else:
                new_pos = pos + min
                f_idx[fold[i]].extend(samples_idx[pos : new_pos])
            pos = new_pos

    for i in range(3):
            train_folds[fold[i]] = train.loc[f_idx[fold[i]]]




    # Split Train in 3-fold
    # TODO: enforce minimal 1 per Gene_KO --> 3 fold
    # TODO: Bootrstrap with same split
    # for c in range(len(classes)):
    #     current = train.loc[train['label'] == c]
    #     n = round(current.shape[0])
    #     current_idx = list(current.index)
    #     random.shuffle(current_idx)
    #
    #     train_1_idx.extend(random.sample(list(current.index), n))
    #
    # train_1 = train.loc[train_1_idx]
    # train_2 = train.drop(train_1_idx)

    # BOOTSTRAP data to equal class distributions

    train_1 = bootstrap_equal(train_folds[0], N)
    train_2 = bootstrap_equal(train_folds[1], N)
    train_3 = bootstrap_equal(train_folds[2], N)

    train = [train_1, train_2, train_3]

    test = bootstrap_equal(test, N)

    # X1_train = train_1.loc[:,features].values
    # y1_train = train_1.loc[:, "label"].values
    # X2_train = train_2.loc[:,features].values
    # y2_train = train_2.loc[:, "label"].values
    #
    # X_test = test.loc[:,features].values
    # y_test = test.loc[:, "label"].values

    return train, test


def data_split_new(N_all, nested):
    '''

    :param N:
    :param nested: Boolean
    :return:
    '''
    data = pd.read_pickle("data/zou2021/zou_96SBS_filtered.pkl")
    features = data.columns[0:96]
    genes = data['Gene_KO'].unique().tolist()
    n_samples = data.size

    # Control
    c0 = ['ATP2B4']
    # MMR
    c1 = ['MLH1', 'MSH2', 'MSH6', 'PMS2', 'PMS1']
    # HR
    c2 = ['EXO1', 'RNF168']
    # BER
    c3 = ['OGG1', 'UNG']

    classes = [c0, c1, c2, c3]

    data["label"] = -1
    for index, row in data.iterrows():
        for i in range(len(classes)):
            if row['Gene_KO'] in classes[i]:
                data.loc[index, "label"] = i
                break

    temp = data.loc[data['label'] == -1]

    test_idx = []

    # outer Test Set
    if nested:
        for g in genes:
            # Number of samples in Gene_KO
            samples = list(data.loc[data["Gene_KO"] == g].index)
            # for control cell line -> 2 test samples
            if g == 'ATP2B4':
                test_idx.extend(random.sample(samples, 2))
            else:
                test_idx.extend(random.sample(samples, 1))


    test = data.loc[test_idx]
    train = data.drop(test_idx)


    f_idx = [[] for i in range(3)]
    train_folds = [None] * 3

    for g in genes:
        # Split remaining Samples evenly over folds
        # If not divisable by 3 -> remaining 1 or 2 in random fold

        samples = train.loc[train["Gene_KO"] == g]
        samples_idx = list(samples.index)
        random.shuffle(samples_idx)
        n = len(samples_idx)

        min = math.floor(n / 3)
        mod = n % 3
        fold = [0, 1, 2]
        random.shuffle(fold)

        pos = 0
        for i in range(len(fold)):
            if mod > 0:
                new_pos = pos + min + 1
                f_idx[fold[i]].extend(samples_idx[ pos: new_pos ])
                mod -= 1
            else:
                new_pos = pos + min
                f_idx[fold[i]].extend(samples_idx[pos : new_pos])
            pos = new_pos

    for i in range(3):
            train_folds[fold[i]] = train.loc[f_idx[fold[i]]]


    data_final = []

    for N in N_all:

        train_1 = bootstrap_equal(train_folds[0], N)
        train_2 = bootstrap_equal(train_folds[1], N)
        train_3 = bootstrap_equal(train_folds[2], N)

        train = [train_1, train_2, train_3]

        test_N = bootstrap_equal(test, N)

        data_final.append((train, test_N))


    return data_final



N = [10,50,100,200,500,1000,2500]

data_all = data_split_new(N,True)


# N = 20
# train, test = data_split_new(N, True)
# pass

# run_MLR(X_train, X_test, y_train, y_test)


#
# print('Predicted value is =', lr.predict([X_test[0]]))
#
# y_pred = lr.predict(X_test)
#
# conf = metrics.confusion_matrix(y_test, y_pred)
#
# #Creating matplotlib axes object to assign figuresize and figure title
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title('Confusion Matrx')
#
# print(metrics.classification_report(y_test, lr.predict(X_test)))
#
# # disp = metrics.plot_confusion_matrix(lr, X_test, y_test, ax = ax)
# # disp.confusion_matrix
# # plt.show()
#
