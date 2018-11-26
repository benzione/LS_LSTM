from threading import Thread
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from util import points_dict
from sklearn.metrics import confusion_matrix
import os
import re

k1 = 3000
k2 = 14
k3 = 5


def run_nb(x_train, y_train, x_test, y_test, parameters, str_file1):
    results = np.zeros(1)
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, int(parameters[0][0])))),
                         ('tfidf', TfidfTransformer(use_idf=int(parameters[0][1]))),
                         ('clf', MultinomialNB()),
                         ])
    text_clf.fit(x_train, y_train)
    predicted = text_clf.predict(x_test)
    np.savetxt('./Data/Results_all/nb_%s.csv' % (str_file1), predicted, delimiter=",")
    # conf = confusion_matrix(y_test, predicted)
    # np.save('./Data/Results_all/nb_%s.npy' % str_file1, conf)


def run_svm(x_train, y_train, x_test, y_test, parameters, function, str_file1):
    results = np.zeros(1)
    if function == 'linear':
        i = 1
    elif function == 'rbf':
        i = 2
    else:
        i = 3
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, int(parameters[i][0])))),
                         ('tfidf', TfidfTransformer(use_idf=int(parameters[i][1]))),
                         ('clf', svm.SVC(kernel=function, C=parameters[i][2], random_state=3)),
                         ])
    text_clf.fit(x_train, y_train)
    predicted = text_clf.predict(x_test)
    np.savetxt('./Data/Results_all/svm_%s_%s.csv' % (function, str_file1), predicted, delimiter=",")
    # conf = confusion_matrix(y_test, predicted)
    # np.save('./Data/Results_all/svm_%s_%s.npy' % (function, str_file1), predicted)


def run_svc(x_train, y_train, x_test, y_test, parameters, str_file1):
    results = np.zeros(1)
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, int(parameters[4][0])))),
                         ('tfidf', TfidfTransformer(use_idf=int(parameters[4][1]))),
                         ('clf', svm.LinearSVC(C=parameters[4][2], random_state=3)),
                         ])
    text_clf.fit(x_train, y_train)
    predicted = text_clf.predict(x_test)
    np.savetxt('./Data/Results_all/svc_%s.csv' % (str_file1), predicted, delimiter=",")


def run_threads(x_train, y_train, x_test, y_test, parameters, flag, str_file1):
    results = np.zeros(k3)
    if flag:
        thread1 = Thread(target=run_nb, args=(x_train, y_train, x_test, y_test, parameters, str_file1))
    thread2 = Thread(target=run_svm, args=(x_train, y_train, x_test, y_test, parameters, 'linear', str_file1))
    thread3 = Thread(target=run_svm, args=(x_train, y_train, x_test, y_test, parameters, 'rbf', str_file1))
    thread4 = Thread(target=run_svm, args=(x_train, y_train, x_test, y_test, parameters, 'poly', str_file1))
    thread5 = Thread(target=run_svc, args=(x_train, y_train, x_test, y_test, parameters, str_file1))
    if flag:
        thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    if flag:
        thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()


def run_test(trajectories_traininig, trajectories_testing, ls_training, ls_testing, parameters, str_file1):
    x_train = trajectories_traininig
    y_train = ls_training

    x_test = trajectories_testing
    y_test = ls_testing
    run_threads(x_train, y_train, x_test, y_test, parameters, k3==5, str_file1)


def main():
    unique_trajectories = np.load('./Data/trajectories_uniques.npy')

    lst_labels = []
    with open('./Data/Labels/labels_uniques_%d.txt' % k1, 'r') as f:
        for line in f:
            tmp = line.strip('\n')
            lst_labels.append(int(tmp))

    lst_labels_ls = []
    str_file = './Data/clusters_trajectories/labels_%d_%d.txt' % (k1, k2)
    with open(str_file, 'r') as f:
        for line in f:
            tmp = line.strip('\n')
            lst_labels_ls.append(int(tmp))

    items = []
    with open('./Data/cross_validation.txt', 'r') as f:
        for line in f:
            tmp = line.strip('\n')
            items.append(int(tmp))

    lst_labels_true = []
    for label in lst_labels:
        lst_labels_true.append(lst_labels_ls[label])

    trajectory_dict = {}
    for i, trajectory in enumerate(unique_trajectories):
        tmp = ''
        for stop_area in trajectory:
            tmp += str(int(stop_area)) + ' '
        tmp = tmp[:-1]
        trajectory_dict[tmp] = lst_labels_true[i]

    names_dict, label_list = points_dict()
    n = len(names_dict)
    results_cv = np.zeros([n, k3])
    results = np.load('./Data/Results_RNN/svm_results.npy')
    parameters = np.zeros([k3, 3])
    max_results = np.argmax(results, axis=0)
    for i, idx in enumerate(max_results):
        tmp = np.load('./Data/Results_RNN/svm_parameters_cv%d.npy' % idx)
        parameters[i,] = tmp[i,]

    for i, file_tmp in enumerate(os.listdir("./Data/Trajectories")):
        str_file1 = file_tmp.title()
        str_file1 = str_file1[:-4]
        with open('./Data/Trajectories/%s.txt' % (str_file1), 'r') as f:
            ls_testing = []
            trajectories_testing = []
            for line in f:
                lst = re.split(r'[ ]', line)
                lst.pop(-1)
                tmp = ''
                for stop_area in lst:
                    tmp += str(int(float(stop_area))) + ' '
                tmp = tmp[:-1]
                trajectories_testing.append(tmp)
                ls_testing.append(trajectory_dict[tmp])

        ls_training= []
        trajectories_traininig = []
        for file_tmp in os.listdir("./Data/Trajectories"):
            str_file2 = file_tmp.title()
            str_file2 = str_file2[:-4]
            if str_file1 != str_file2:
                with open('./Data/Trajectories/%s.txt' % (str_file2), 'r') as f:
                    for line in f:
                        lst = re.split(r'[ ]', line)
                        lst.pop(-1)
                        tmp = ''
                        for stop_area in lst:
                            tmp += str(int(float(stop_area))) + ' '
                        tmp = tmp[:-1]
                        trajectories_traininig.append(tmp)
                        ls_training.append(trajectory_dict[tmp])

        print('run test svm %s %d' % (str_file1, i))
        run_test(trajectories_traininig, trajectories_testing, ls_training, ls_testing, parameters, str_file1)
    # np.save('./Data/Results_RNN/svm_results_users.npy', results_cv)


if __name__ == '__main__':
    main()
