import numpy as np
from my_hmm import MultinomialHMM
from util import points_dict
import os
import re
from sklearn.metrics import confusion_matrix
top_words = 36
max_review_length = 24
k1 = 3000
k2 = 14
k3 = 1


def run_model(X_train, train_len, parameters):
    np.random.seed(3)
    model = MultinomialHMM(n_components=int(parameters[0]), n_features=top_words, startprob_prior=1, transmat_prior=1,
                           tol=parameters[1])
    model.fit(X_train, train_len)
    return model


def predict_set(models, X_validate, y_validate, str_file1):
    n = len(y_validate)
    predicted = []
    for i, trajectory in enumerate(X_validate):
        result = np.zeros([1, k2])
        for j, model in enumerate(models):
            result[0, j] = model.score(np.atleast_2d(trajectory).T)
        predicted.append(np.argmax(result))

    np.savetxt('./Data/Results_all/hmm_%s.csv' % (str_file1), np.array(predicted), delimiter=",")

def run_optimize(cv, text, items, lst_labels_true, parameters):
    n = np.max(items)
    results = np.zeros([n, k3])
    i = 0
    for train_valid in range(n + 1):
        if cv != train_valid:
            X_train = [text[j] for j, val in enumerate(items) if val != train_valid and val != cv]
            y_train = [lst_labels_true[j] for j, val in enumerate(items) if val != train_valid and val != cv]

            X_validate = [text[j] for j, val in enumerate(items) if val == train_valid]
            y_validate = [lst_labels_true[j] for j, val in enumerate(items) if val == train_valid]

            models = []
            for ii in range(k2):
                x_train_tmp = [X_train[j] for j, val in enumerate(y_train) if val == ii]
                if x_train_tmp:
                    tmp = np.concatenate(x_train_tmp)
                    tmp = np.reshape(tmp, [-1, 1])
                    training = np.array(tmp).T
                    train_lengths = np.fromiter(map(len, x_train_tmp), int)
                    models.append(run_model(training.T, train_lengths, parameters[0]))
            results[i, 0] = predict_set(models, X_validate, y_validate)
            print('cv %d, iteration %d, and accuracy = %.2f' % (cv, i, results[i, 0]))
            i += 1
    return np.mean(results, axis=0)


def run_test(trajectories_traininig, trajectories_testing, ls_training, ls_testing, parameters, str_file1):
    X_train = trajectories_traininig
    y_train = ls_training

    X_validate = trajectories_testing
    y_validate = ls_testing

    models = []
    for ii in range(k2):
        x_train_tmp = [X_train[j] for j, val in enumerate(y_train) if val == ii]
        if x_train_tmp:
            tmp = np.concatenate(x_train_tmp)
            tmp = np.reshape(tmp, [-1, 1])
            training = np.array(tmp).T
            train_lengths = np.fromiter(map(len, x_train_tmp), int)
            models.append(run_model(training.T, train_lengths, parameters[0]))
    predict_set(models, X_validate, y_validate, str_file1)


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
        tmp = []
        for stop_area in trajectory:
            tmp.append(int(float(stop_area)))
        trajectory_dict[repr(tmp)] = lst_labels_true[i]

    names_dict, label_list = points_dict()
    n = len(names_dict)
    results_cv = np.zeros([n, 1])
    results = np.load('./Data/Results_RNN/hmm_results.npy')
    max_results = np.argmax(results)
    parameters = np.load('./Data/Results_RNN/hmm_parameters_cv%d.npy' % max_results)

    for i, file_tmp in enumerate(os.listdir("./Data/Trajectories")):
        str_file1 = file_tmp.title()
        str_file1 = str_file1[:-4]
        with open('./Data/Trajectories/%s.txt' % (str_file1), 'r') as f:
            ls_testing = []
            trajectories_testing = []
            for line in f:
                lst = re.split(r'[ ]', line)
                lst.pop(-1)
                tmp = []
                for stop_area in lst:
                    tmp.append(int(float(stop_area)))
                trajectories_testing.append(tmp)
                ls_testing.append(trajectory_dict[repr(tmp)])

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
                        tmp = []
                        for stop_area in lst:
                            tmp.append(int(float(stop_area)))
                        trajectories_traininig.append(tmp)
                        ls_training.append(trajectory_dict[repr(tmp)])

        print('run test hmm %s %d' % (str_file1, i))
        run_test(trajectories_traininig, trajectories_testing, ls_training, ls_testing, parameters, str_file1)


if __name__ == '__main__':
    main()
