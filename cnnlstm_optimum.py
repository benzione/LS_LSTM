import numpy as np
import gc
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from multiprocessing import Process, Queue
from keras.utils.np_utils import to_categorical
from keras import backend as K
from util import points_dict
import os
import re
from sklearn.metrics import confusion_matrix
top_words = 36
max_review_length = 24
k1 = 3000
k2 = 14
k3 = 1


def run_in_separate_process(method, args):
    def queue_wrapper(q, params):
        r = method(*params)
        q.put(r)

    q = Queue()
    p = Process(target=queue_wrapper, args=(q, args))
    p.start()
    return_val = q.get()
    p.join()
    return return_val


def run_model(X_train, y_train, X_validate, y_validate, parameters, str_file1):
    np.random.seed(3)

    model = Sequential()
    model.add(Embedding(top_words, int(parameters[0]), input_length=max_review_length))
    model.add(Dropout(float(parameters[1])))
    model.add(Convolution1D(filters=int(parameters[2]),
                            kernel_size=int(parameters[3]),
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(MaxPooling1D(pool_size=int(parameters[4])))
    model.add(Dropout(float(parameters[5])))
    model.add(LSTM(int(parameters[6])))
    model.add(Dropout(float(parameters[7])))
    model.add(Dense(k2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, to_categorical(y_train), epochs=int(parameters[8]), batch_size=int(parameters[9]), verbose=0,
              shuffle=True)

    # Final evaluation of the model
    # scores = model.evaluate(X_validate, to_categorical(y_validate), verbose=0)
    #
    # model = X_train = y_train = X_validate = y_validate = parameters = None
    # gc.collect()
    # return scores[1] * 100
    predicted = model.predict(X_validate, verbose=0)
    np.savetxt('./Data/Results_all/cnnlstm_%s.csv' % (str_file1), predicted, delimiter=",")
    # conf = confusion_matrix(y_validate, predicted)
    # np.save('./Data/Results_all/cnnlstm_%s.npy' % str_file1, conf)


def run_optimize(cv, text, items, lst_labels_true, parameters):
    n = np.max(items)
    i = 0
    results = np.zeros([n, k3])
    for train_valid in range(n + 1):
        if cv != train_valid:
            X_train = [text[j] for j, val in enumerate(items) if val != train_valid and val != cv]
            y_train = [lst_labels_true[j] for j, val in enumerate(items) if val != train_valid and val != cv]

            X_validate = [text[j] for j, val in enumerate(items) if val == train_valid]
            y_validate = [lst_labels_true[j] for j, val in enumerate(items) if val == train_valid]

            # print("Accuracy: %.2f%%" % (scores[1]*100))
            results[i, 0] = run_model(X_train, y_train, X_validate, y_validate, parameters[0])
            print('cv %d, iteration %d, and result = %.2f' % (cv, i, results[i, 0]))
            i += 1
            X_train = y_train = X_validate = y_validate = None
    cv = text = items = lst_labels_true = parameters = n = i = None
    K.clear_session()
    gc.collect()
    return np.mean(results, axis=0)


def run_test(trajectories_traininig, trajectories_testing, ls_training, ls_testing, parameters, str_file1):
    X_train = trajectories_traininig
    y_train = ls_training

    X_validate = trajectories_testing
    y_validate = ls_testing

    run_model(X_train, y_train, X_validate, y_validate, parameters[0], str_file1)
    K.clear_session()
    # return results


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
    results = np.load('./Data/Results_RNN/cnnlstm_results.npy')
    max_results = np.argmax(results)
    parameters = np.load('./Data/Results_RNN/cnnlstm_parameters_cv%d.npy' % max_results)

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

        print('run test cnnlstm %s %d' % (str_file1, i))
        run_test(trajectories_traininig, trajectories_testing, ls_training, ls_testing, parameters, str_file1)
    # np.save('./Data/Results_RNN/cnnlstm_results_users.npy', results_cv)


if __name__ == '__main__':
    main()
