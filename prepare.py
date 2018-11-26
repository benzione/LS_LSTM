import json
import math
import os
import re
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import AgglomerativeClustering as agclus
from sklearn import metrics
import numpy_indexed as npi
import simplekml
from util import points_dict
from multiprocessing import Pool
# import multiprocessing
import gensim
import sys
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools


def round_lat_long(data, d):
    rc = 10000
    dd = d / float(rc)
    return np.round(data * rc / d) * dd


def dt_parse(t):
    ret = datetime.strptime(t[0:19], '%Y-%m-%d %H:%M:%S')
    if t[19] == '+':
        ret += timedelta(hours=int(t[20:22]), minutes=int(t[23:25]))
    elif t[19] == '-':
        ret -= timedelta(hours=int(t[20:22]), minutes=int(t[23:25]))
    return ret


def levenshtein(s1, s2):
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def dtw(a, b):
    # create the cost matrix
    m, n = len(a) + 1, len(b) + 1
    distance = np.zeros([m, n])

    # initialize the first row and column
    for i in range(1, m):
        distance[i, 0] = np.inf

    for j in range(1, n):
        distance[0, j] = np.inf

    # fill in the rest of the matrix
    for i in range(1, m):
        for j in range(1, n):
            cost = abs(a[i - 1] - b[j - 1])
            distance[i, j] = cost + min(distance[i - 1, j], distance[i, j - 1], distance[i - 1, j - 1])

    return distance[-1, -1]


def dunn(c, distances):
    unique_cluster_distances = np.unique(min_cluster_distances(c, distances))
    max_diameter = max(diameter(c, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(c, distances):
    """Calculates the distances between the two nearest points of each cluster"""
    min_distances = np.zeros((max(c) + 1, max(c) + 1))
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            if c[i] != c[ii] and distances[i, ii] > min_distances[c[i], c[ii]]:
                min_distances[c[i], c[ii]] = min_distances[c[ii], c[i]] = distances[i, ii]
    return min_distances


def diameter(c, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)"""
    diameters = np.zeros(max(c) + 1)
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            if c[i] != -1 or c[ii] != -1 and c[i] == c[ii] and distances[i, ii] > diameters[c[i]]:
                diameters[c[i]] = distances[i, ii]

    return diameters


def process_json():
    for file_tmp in os.listdir("/home/eyal/Algorithms/LMP/Location Google"):
        if file_tmp.endswith(".json"):
            str_file = file_tmp.title()
            str_file_new = './Data/Raw/%s.txt' % (str_file[:-5])
            with open("/home/eyal/Algorithms/LMP/Location Google/%s.json" % (str_file[:-5])) as data_file:
                data_json = json.load(data_file)
            data_file.close()
            print(str_file_new)
            data_raw = data_json['locations']
            the_file = open(str_file_new, 'w')
            for i in range(len(data_raw)-1, -1, -1):
                lat = data_raw[i]['latitudeE7']/float(10000000)
                lon = data_raw[i]['longitudeE7']/float(10000000)
                ts = math.floor(int(data_raw[i]['timestampMs'])/1000)
                utc_dt = datetime.fromtimestamp(ts)
                coordinates = round_lat_long(np.array([lat, lon]), 5)
                the_file.write("%s,%s,%s\n" % (str(coordinates[0]), str(coordinates[1]), str(utc_dt)))
            the_file.close()


def find_blanks():
    for file_tmp in os.listdir("./Data/Raw"):
        if file_tmp.endswith(".txt"):
            str_file = file_tmp.title()
            str_file_pre = './Data/Raw/%s.txt' % (str_file[:-4])
            str_file_new = './Data/WithOutBlank/%s.txt' % (str_file[:-4])
            with open(str_file_pre, 'r') as f:
                print(str_file)
                pre = ''
                the_file = open(str_file_new, 'w')
                data_list = []
                data_list_week = []
                flag = 1
                count = 1
                delta = 0
                ts = 0
                sunday_pre_week = 0
                for line in f:
                    line = line.strip('\n')
                    data = re.split(r'[,]', line)
                    ts = datetime.strptime(data[2][:-1][0:19], '%Y-%m-%d %H:%M:%S')
                    if pre != '':
                        if (ts - sunday_pre).days >= 14:
                            sunday_pre = ts - timedelta(days=(ts.weekday() + 1) % 7, hours=ts.hour,
                                                        minutes=ts.minute, seconds=ts.second)
                            tmp = (sunday_pre - pre).days * (24 * 60 * 60) + (sunday_pre - pre).seconds
                            delta += math.floor(tmp / float(60 * 60))
                            pre = sunday_pre
                            if delta <= 24:
                                for list_r in data_list:
                                    the_file.write("%s,%s,%s,%d\n" % (list_r[0], list_r[1], list_r[2], count))

                                tmp = sunday_pre
                                sunday_pre = sunday_pre_week
                                sunday_pre_week = tmp
                                data_list = data_list_week
                                data_list_week = [data]
                                count += 1
                            else:
                                data_list = []
                                data_list_week = []
                                flag = 1

                            tmp = (ts - pre).days * (24 * 60 * 60) + (ts - pre).seconds
                            delta = math.floor(tmp / float(60 * 60))
                        elif (ts - sunday_pre).days >= 7:
                            if flag:
                                sunday_pre_week = ts - timedelta(days=(ts.weekday() + 1) % 7, hours=ts.hour,
                                                                 minutes=ts.minute, seconds=ts.second)
                                tmp = (sunday_pre_week - pre).days * (24 * 60 * 60) + (sunday_pre_week - pre).seconds
                                delta += math.floor(tmp / float(60 * 60))
                                if delta <= 24:
                                    data_list_week.append(data)
                                    flag = 0
                                else:
                                    data_list = []
                                    data_list_week = []
                                    sunday_pre = sunday_pre_week
                                pre = sunday_pre_week
                                tmp = (ts - pre).days * (24 * 60 * 60) + (ts - pre).seconds
                                delta = math.floor(tmp / float(60 * 60))

                            else:
                                tmp = (ts - pre).days * (24 * 60 * 60) + (ts - pre).seconds
                                delta += math.floor(tmp / float(60 * 60))
                                data_list_week.append(data)

                        else:
                            tmp = (ts - pre).days * (24 * 60 * 60) + (ts - pre).seconds
                            delta += math.floor(tmp / float(60 * 60))
                    else:
                        sunday_pre = ts - timedelta(days=(ts.weekday() + 1) % 7, hours=ts.hour, minutes=ts.minute, seconds=ts.second)
                        pre = sunday_pre
                        tmp = (ts - pre).days * (24 * 60 * 60) + (ts - pre).seconds
                        delta = math.floor(tmp / float(60 * 60))
                    pre = ts
                    data_list.append(data)

                sunday_next = sunday_pre + timedelta(days=14)
                tmp = (sunday_next - pre).days * (24 * 60 * 60) + (sunday_next - pre).seconds
                delta += math.floor(tmp / float(60 * 60))
                if delta <= 24:
                    for list_r in data_list:
                        the_file.write("%s,%s,%s,%d\n" % (list_r[0], list_r[1], list_r[2], count))
                the_file.close()
            f.close()

def find_coordinate(count_arr, data_dict):
    if count_arr[0]:
        key_best = np.argmax(count_arr)
        for key, val in data_dict.items():
            if key_best == val:
                return key

def prepare_to_input():
    for file_tmp in os.listdir("./Data/WithOutBlank"):
        if file_tmp.endswith(".txt"):
            str_file = file_tmp.title()
            str_file_pre = './Data/WithOutBlank/%s.txt' % (str_file[:-4])
            if os.stat(str_file_pre).st_size:
                with open(str_file_pre, 'r') as f:
                    print(str_file)
                    pre = ''
                    data_dict = {}
                    # count_dict = {}
                    count_arr = np.zeros(10000)
                    i = 0
                    coordinates_list = []
                    trajectory_pre = 0
                    trajectory_now = 0
                    for line in f:
                        line = line.strip('\n')
                        data = re.split(r'[,]', line)
                        # ts = datetime.strptime(data[2][:-1][0:19], '%Y-%m-%d %H:%M:%S')
                        ts = datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')
                        trajectory_now = data[3]
                        if pre != '':
                            if (ts - hour_pre).seconds >= (60 * 60):
                                coordinates = find_coordinate(count_arr, data_dict)

                                coordinates_list.append(str([coordinates, (hour_pre.weekday() + 1) % 7, hour_pre.hour,
                                                             trajectory_pre]))
                                trajectory_pre = trajectory_now
                                data_dict = {}
                                # count_dict = {}
                                count_arr = np.zeros(10000)
                                i = 0
                                hour_pre = ts - timedelta(minutes=ts.minute, seconds=ts.second)
                        else:
                            hour_pre = ts - timedelta(minutes=ts.minute, seconds=ts.second)
                            trajectory_pre = trajectory_now

                        pre = ts
                        str_dict = str([data[0], data[1]])
                        val = data_dict.get(str_dict, "")
                        if val != "":
                            count_arr[val] += 1
                        else:
                            data_dict[str_dict] = i
                            # count_dict[str(i)] = 1
                            count_arr[i] = 1
                            i += 1

                    coordinates = find_coordinate(count_arr, data_dict)

                    if coordinates:
                        coordinates_list.append(str([coordinates, (hour_pre.weekday() + 1) % 7, hour_pre.hour,
                                                     trajectory_now]))

                    str_file_new = './Data/CoordinatesInput/%s.txt' % (str_file[:-4])
                    the_file = open(str_file_new, 'w')
                    for list_tmp in coordinates_list:
                        the_file.write("%s\n" % list_tmp)
                    the_file.close()
                f.close()


def create_kml():
    for file_tmp in os.listdir("./Data/CoordinatesInput"):
        if file_tmp.endswith(".txt"):
            str_file = file_tmp.title()
            str_file_pre = './Data/CoordinatesInput/%s.txt' % (str_file[:-4])
            str_file_kml = './Data/KML/%s.kml' % (str_file[:-4])
            with open(str_file_pre, 'r') as f:
                print(str_file)
                all_coordinates = []
                for line in f:
                    line = line.strip('\n')
                    tmp = ''
                    coordinates_list = []
                    for sign in line:
                        if sign != '[' and sign != ']' and sign != "'" and sign != '"' and sign != ' ':
                            if sign == ',':
                                coordinates_list.append(tmp)
                                tmp = ''
                            else:
                                tmp += sign
                    coordinates_list.append(tmp)
                    all_coordinates.append(coordinates_list)

            if len(all_coordinates) > 0:
                data = np.zeros([len(all_coordinates), 6])
                for idx, item1 in enumerate(all_coordinates):
                    for idy, item2 in enumerate(item1):
                        data[idx][idy] = float(item2)

                tmp = data[:, :2]
                unique_coordinate = npi.unique(tmp)
                label = npi.indices(unique_coordinate, tmp)
                max_cluster = np.max(label) + 1
                count = np.zeros(max_cluster)
                for i in label:
                    count[i] += 1

                temp = count
                idc = np.zeros(max_cluster)
                for i in range(max_cluster):
                    idx = np.argmax(temp)
                    idc[idx] = max_cluster - (i + 1)
                    temp[idx] = 0

                for idx, count_label in enumerate(idc):
                    if count_label < max_cluster - math.floor(max_cluster * 0.1):
                        label[np.where(label == idx)] = max_cluster

                j = 0
                for i in range(max_cluster + 1):
                    tmp = np.where(label == i)
                    if len(tmp[0]) > 0:
                        label[tmp] = j
                        j += 1

                max_cluster = np.max(label) + 1
                kml = simplekml.Kml()
                for i in range(max_cluster):
                    tmp = np.where(label == i)
                    if tmp:
                        kml.newpoint(name=str(i+1), coords=[(data[tmp[0][0], 1], data[tmp[0][0], 0])])
                kml.save(str_file_kml)


def create_np():
    names_dict, label_list = points_dict()
    for file_tmp in os.listdir("./Data/CoordinatesInput"):
        if file_tmp.endswith(".txt"):
            str_file = file_tmp.title()
            str_file_pre = './Data/CoordinatesInput/%s.txt' % (str_file[:-4])
            str_file_np = './Data/npData/%s' % (str_file[:-4])
            with open(str_file_pre, 'r') as f:
                print(str_file)
                all_coordinates = []
                for line in f:
                    line = line.strip('\n')
                    tmp = ''
                    coordinates_list = []
                    for sign in line:
                        if sign != '[' and sign != ']' and sign != "'" and sign != '"' and sign != ' ':
                            if sign == ',':
                                coordinates_list.append(tmp)
                                tmp = ''
                            else:
                                tmp += sign
                    coordinates_list.append(tmp)
                    all_coordinates.append(coordinates_list)

            if len(all_coordinates) > 0:
                data = np.zeros([len(all_coordinates), 6])
                for idx, item1 in enumerate(all_coordinates):
                    for idy, item2 in enumerate(item1):
                        data[idx][idy] = float(item2)

                tmp = data[:, :2]
                unique_coordinate = npi.unique(tmp)
                label = npi.indices(unique_coordinate, tmp)
                max_cluster = np.max(label) + 1
                count = np.zeros(max_cluster)
                for i in label:
                    count[i] += 1

                temp = count
                idc = np.zeros(max_cluster)
                for i in range(max_cluster):
                    idx = np.argmax(temp)
                    idc[idx] = max_cluster - (i + 1)
                    temp[idx] = 0

                for idx, count_label in enumerate(idc):
                    if count_label < max_cluster - math.floor(max_cluster * 0.1):
                        label[np.where(label == idx)] = max_cluster

                j = 0
                for i in range(max_cluster + 1):
                    tmp = np.where(label == i)
                    if len(tmp[0]) > 0:
                        data[np.where(label == i), 5] = label_list[names_dict[str_file[:-4]]][j]
                        j += 1

                np.save(str_file_np, data)


def prepare_trajectory():
    max_stop = 35
    # trajectory_list = []
    for file_tmp in os.listdir("./Data/npData"):
        trajectory_list = []
        if file_tmp.endswith(".npy"):
            count = 0
            pre_sentence = 0
            str_file = file_tmp.title()
            str_file_pre = './Data/npData/%s.npy' % (str_file[:-4])
            data = np.load(str_file_pre)
            print(str_file)
            trajectory = []
            if data[0, 3] > 0:
                tmp1 = data[0, 3]
                id_record = 0
                while id_record < tmp1:
                    trajectory.append(max_stop)
                    id_record += 1
            trajectory.append(data[0, -1])
            for i in range(1, len(data)):
                if data[i, 3] < data[i-1, 3]:
                    tmp1 = data[i, 3] + 24
                else:
                    tmp1 = data[i, 3]

                if tmp1 - data[i-1, 3] > 1:
                    id_record = data[i-1, 3] + 1
                    while id_record < tmp1:
                        if id_record % 24 == 0:
                            if pre_sentence != data[i-1, 4]:
                                pre_sentence = data[i-1, 4]
                            trajectory_list.append(trajectory)
                            trajectory = []

                        trajectory.append(max_stop)
                        id_record += 1

                    if tmp1 % 24 == 0:
                        if pre_sentence != data[i-1, 4]:
                            pre_sentence = data[i-1, 4]
                        trajectory_list.append(trajectory)
                        trajectory = []
                    trajectory.append(data[i, -1])
                else:
                    if tmp1 % 24 == 0:
                        if pre_sentence != data[i-1, 4]:
                            # count += 1
                            pre_sentence = data[i-1, 4]
                        trajectory_list.append(trajectory)
                        trajectory = []
                    trajectory.append(data[i, -1])

            while len(trajectory) < 24:
                    trajectory.append(max_stop)
            trajectory_list.append(trajectory)

        str_file_pre = './Data/Trajectories/%s.txt' % (str_file[:-4])
        # str_file_pre = './Data/Trajectories.txt'
        f = open(str_file_pre, 'w')
        for inner_list in trajectory_list:
            for element in inner_list:
                f.write(str(element) + ' ')
            f.write('\n')
        f.close()


def prepare_trajectory_all():
    max_stop = 35
    trajectory_list = []
    for file_tmp in os.listdir("./Data/npData"):
        # trajectory_list = []
        if file_tmp.endswith(".npy"):
            count = 0
            pre_sentence = 0
            str_file = file_tmp.title()
            str_file_pre = './Data/npData/%s.npy' % (str_file[:-4])
            data = np.load(str_file_pre)
            print(str_file)
            trajectory = []
            if data[0, 3] > 0:
                tmp1 = data[0, 3]
                id_record = 0
                while id_record < tmp1:
                    trajectory.append(max_stop)
                    id_record += 1
            trajectory.append(data[0, -1])
            for i in range(1, len(data)):
                if data[i, 3] < data[i-1, 3]:
                    tmp1 = data[i, 3] + 24
                else:
                    tmp1 = data[i, 3]

                if tmp1 - data[i-1, 3] > 1:
                    id_record = data[i-1, 3] + 1
                    while id_record < tmp1:
                        if id_record % 24 == 0:
                            if pre_sentence != data[i-1, 4]:
                                pre_sentence = data[i-1, 4]
                            trajectory_list.append(trajectory)
                            trajectory = []

                        trajectory.append(max_stop)
                        id_record += 1

                    if tmp1 % 24 == 0:
                        if pre_sentence != data[i-1, 4]:
                            pre_sentence = data[i-1, 4]
                        trajectory_list.append(trajectory)
                        trajectory = []
                    trajectory.append(data[i, -1])
                else:
                    if tmp1 % 24 == 0:
                        if pre_sentence != data[i-1, 4]:
                            # count += 1
                            pre_sentence = data[i-1, 4]
                        trajectory_list.append(trajectory)
                        trajectory = []
                    trajectory.append(data[i, -1])

            while len(trajectory) < 24:
                    trajectory.append(max_stop)
            trajectory_list.append(trajectory)

    # str_file_pre = './Data/Trajectories/%s.txt' % (str_file[:-4])
    str_file_pre = './Data/Trajectories.txt'
    f = open(str_file_pre, 'w')
    for inner_list in trajectory_list:
        for element in inner_list:
            f.write(str(element) + ' ')
        f.write('\n')
    f.close()


def create_pre_compute():
    with open('./Data/Trajectories.txt', 'r') as f:
        # trajectory_num = []
        trajectories_list = []
        for line in f:
            lst = re.split(r'[ ]', line)
            lst.pop(-1)
            tmp = [float(i) for i in lst]
            # trajectory_num.append(tmp.pop(-1))
            trajectories_list.append(tmp)

    trajectories = np.zeros([len(trajectories_list), 24])
    for i, trajectory in enumerate(trajectories_list):
        for j, stop_area in enumerate(trajectory):
            trajectories[i, j] = int(stop_area)

    unique_trajectories = npi.unique(trajectories)
    label_uniques = npi.indices(unique_trajectories, trajectories)
    np.save('./Data/label_uniques.npy', label_uniques)
    np.save('./Data/trajectories_uniques.npy', unique_trajectories)
    np.save('./Data/np_trajectories.npy', trajectories)

    # trajectories = np.load('./Data/np_trajectories.npy')
    # unique_trajectories = np.load('./Data/trajectories_uniques.npy')
    # label_uniques = np.load('./Data/label_uniques.npy')

    dist_pre_compute = np.zeros([len(unique_trajectories), len(unique_trajectories)])
    for i in range(0, len(unique_trajectories)-1):
        for j in range(i+1, len(unique_trajectories)):
            if j % 2000 == 0:
                print([i, j])
            dist_pre_compute[i, j] = dist_pre_compute[j, i] = levenshtein(unique_trajectories[i, :], unique_trajectories[j, :])
            # dist_pre_compute[i, j] = dist_pre_compute[j, i] = dtw(unique_trajectories[i, :], unique_trajectories[j, :])
    np.save('./Data/pre_compute_dtw', dist_pre_compute)


def cluster_trajectories(k):
    # k *= 20
    dist_pre_compute = np.load('./Data/pre_compute_dtw.npy')

    model = agclus(linkage='complete', affinity="precomputed", n_clusters=k)
    model.fit(dist_pre_compute)

    label_uniques = np.load('./Data/label_uniques.npy')
    tmp = model.labels_[label_uniques]

    str_file_new = './Data/Labels/labels_%d.txt' % k
    the_file = open(str_file_new, 'w')
    for element in tmp:
        the_file.write("%s\n" % element)
    the_file.close()

    str_file_new = './Data/Labels/labels_uniques_%d.txt' % k
    the_file = open(str_file_new, 'w')
    for element in model.labels_:
        the_file.write("%s\n" % element)
    the_file.close()


def run_w2v(k):
    lst_labels = []
    with open('./Data/Labels/labels_%d.txt' % k, 'r') as f:
        for line in f:
            tmp = line.strip('\n')
            lst_labels.append(tmp)
    count = 1
    text = []
    sentence = []
    for element in lst_labels:
        sentence.append(element)
        if count % 14 == 0:
            text.append(sentence)
            sentence = []
            count = 0
        count += 1

    # multiprocessing.cpu_count()
    if k <= 40:
        dim = 10
    elif k <= 100:
        dim = 20
    else:
        dim = 50
    model = gensim.models.Word2Vec(size=dim, iter=500, window=14, workers=10,
                                   negative=1, min_count=0, hs=0, sg=0, alpha=1e-3, min_alpha=1e-3, seed=3)
    model.build_vocab(text)
    gap = (1e-3 - 1e-6) / 30
    for i in range(30):
        model.train(text, total_examples=len(text), epochs=model.iter)
        model.alpha -= gap

    model.init_sims(replace=True)
    model.word2id = dict((w, v.index) for w, v in model.wv.vocab.items())
    model.word_vectors = model.wv.syn0norm

    model.save('./Data/w2v/w2v_%d' % k)


def run_tsne(k):
    model = gensim.models.Word2Vec.load('./Data/w2v/w2v_%d' % k)
    we = model.word_vectors
    model = TSNE(n_components=2, perplexity=5, learning_rate=100,
                 n_iter=5000, n_iter_without_progress=30, random_state=0)
    z = model.fit_transform(we)
    np.save('./Data/TSNE/tsne_%d.npy' % k, z)
    # dim = 10
    # plt.scatter(z[:, 0], z[:, 1], s=dim)
    # plt.show()


def cluster_trajectories_to_lifestyle(a, b):
    data = np.load('./Data/TSNE/tsne_%d.npy' % b)
    model = agclus(linkage='complete', affinity="euclidean", n_clusters=a)
    model.fit(data)
    str_file_new = './Data/clusters_trajectories/labels_%d_%d.txt' % (b, a)
    the_file = open(str_file_new, 'w')
    for element in model.labels_:
        the_file.write("%s\n" % element)
    the_file.close()


def cluster_trajectories_to_lifestyle_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return cluster_trajectories_to_lifestyle(*a_b)


def distances_tsne_output(k):
    data = np.load('./Data/TSNE/tsne_%d.npy' % k)
    d = data.shape[0]
    dist = np.zeros([d, d])
    for i in range(d-1):
        for j in range(i+1, d):
            # dist[i, j] = dist[j, i] = np.linalg.norm(data[i, :] - data[j, :])
            dist[i, j] = dist[j, i] = np.sqrt(np.sum((data[i, :] - data[j, :]) ** 2))

    np.save('./Data/distances_tsne_output/dist_%d.npy' % k, dist)


def dunn_index_calculate(k):
    dist = np.load('./Data/distances_tsne_output/dist_%d.npy' % k)
    score_dunn = np.zeros(c-2)
    for i in range(2, c):
        lst_labels = []
        str_file = './Data/clusters_trajectories/labels_%d_%d.txt' % (k, i)
        with open(str_file, 'r') as f:
            for line in f:
                tmp = line.strip('\n')
                lst_labels.append(int(tmp))
        score_dunn[i-2] = dunn(lst_labels, dist)

    np.save('./Data/dunn/scores_%d.npy' % k, score_dunn)


def create_cross_validation():
    unique_trajectories = np.load('./Data/trajectories_uniques.npy')
    len_data_set = len(unique_trajectories)
    items = [i for i in range(5)] * int(math.ceil(len_data_set / float(5)))
    items = items[:len_data_set]
    random.shuffle(items)
    str_file_new = './Data/cross_validation.txt'
    the_file = open(str_file_new, 'w')
    for element in items:
        the_file.write("%s\n" % element)
    the_file.close()


def create_cross_validation_w2v(k):
    lst_labels = []
    if os.path.isfile('./Data/Labels/labels_%d.txt' % k):
        with open('./Data/Labels/labels_%d.txt' % k, 'r') as f:
            for line in f:
                tmp = line.strip('\n')
                lst_labels.append(tmp)
        count = 1
        text = []
        sentence = []
        for element in lst_labels:
            sentence.append(element)
            if count % 14 == 0:
                text.append(sentence)
                sentence = []
                count = 0
            count += 1

    len_data_set = len(text)
    items = [i for i in range(5)] * int(math.ceil(len_data_set / float(5)))
    items = items[:len_data_set]
    random.shuffle(items)
    str_file_new = './Data/cross_validation_w2v.txt'
    the_file = open(str_file_new, 'w')
    for element in items:
        the_file.write("%s\n" % element)
    the_file.close()


def score_cluster(k):
    data = np.load('./Data/TSNE/tsne_%d.npy' % k)
    arr = range(2, 101, 2)
    score_ch = np.zeros(len(arr))
    for i, c  in enumerate(arr):
        lst_labels = []
        str_file = './Data/clusters_trajectories/labels_%d_%d.txt' % (k, c)
        with open(str_file, 'r') as f:
            for line in f:
                tmp = line.strip('\n')
                lst_labels.append(int(tmp))
        score_ch[i] = metrics.calinski_harabaz_score(data, lst_labels)

    np.save('./Data/calinski_harabaz/scores_%d.npy' % (k), score_ch)


def plot_dunn():
    for i in range(500, 5001, 500):
        # i = 950
        data = np.load('./Data/calinski_harabaz/scores_%d.npy' % i)
        print('k:' + str(i) + ' best:' + str(np.argmax(data[:15])))
        # plt.plot(data)
        # plt.show()

if __name__ == '__main__':
    # process_json()
    # find_blanks()

    # prepare_to_input()
    # create_kml()

    # create_np()

    prepare_trajectory_all()
    create_pre_compute()

    pool = Pool(processes=10)
    pool.map(cluster_trajectories, range(500, 5001, 500))
    pool.close()

    pool = Pool(processes=10)
    pool.map(run_w2v, range(500, 5001, 500))
    pool.close()

    pool = Pool(processes=10)
    pool.map(run_tsne, range(500, 5001, 500))
    pool.close()

    for i, k in enumerate(range(500, 5001, 500)):
        print(i)
        pool = Pool(processes=10)
        pool.map(cluster_trajectories_to_lifestyle_star, zip(range(2, 101, 2), itertools.repeat(k)))
        pool.close()

    print('Distances')
    pool = Pool(processes=10)
    pool.map(distances_tsne_output, range(500, 5001, 500))
    pool.close()

    print('Scores')
    pool = Pool(processes=10)
    pool.map(score_cluster, range(500, 5001, 500))
    pool.close()

    # plot_dunn()

    create_cross_validation()
    create_cross_validation_w2v(1500)

    
    # SIZE = 20
    # plt.rc('font', size=SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
    # plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=SIZE)
    # data = np.load('./Data/calinski_harabaz/scores_3000.npy')
    # print(data)
    # plt.plot(data[:14])
    # plt.ylabel('Calinski Harabaz')
    # plt.xlabel('Number of clusters')
    # plt.show()

    # data = np.load('./Data/TSNE/tsne_3000.npy')
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()
