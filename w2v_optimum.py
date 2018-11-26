import gensim
import numpy as np
import os


def softmax(z):
    # print z.shape
    assert len(z.shape) == 1
    s = np.max(z)
    e_x = np.exp(z - s)
    div = np.sum(e_x)
    return e_x / div


def score(model, testing, training, neg, k):
    word2id = model.word2id
    N = len(training)
    initial_prob = np.zeros(k)
    for it in xrange(N):
        x = training[it]
        initial_prob[word2id[x[0]]] += 1
    initial_prob /= float(np.sum(initial_prob))

    V = model.syn0
    if neg == 20:
        U = model.syn1neg
    else:
        U = model.syn1

    N = len(testing)
    costs = 0
    for it in xrange(N):
        x = testing[it]
        n = len(x)
        for jj in xrange(0, n):
            if jj > 0:
                current = word2id[x[jj]]
                previous = word2id[x[jj - 1]]
                z = np.dot(V[previous, :], U.T)
                y = softmax(z)
                if y[current] > 10**-10:
                    costs += np.log(y[current])
                else:
                    costs += np.log(10**-10)
            else:
                if initial_prob[word2id[x[0]]] > 10**-10:
                    costs += np.log(initial_prob[word2id[x[0]]])
                else:
                    costs += np.log(10**-10)

    return -(costs / float(N))


def run_w2v(text, training, dim, iteration, windows, hs, sg, neg, k):
    model = gensim.models.Word2Vec(size=dim, iter=iteration, window=windows, workers=10,
                                   negative=neg, min_count=0, hs=hs, sg=sg,
                                   alpha=1e-3, min_alpha=1e-3, seed=3)
    model.build_vocab(text)

    gap = (1e-3 - 1e-6) / 30
    for i in xrange(30):
        model.train(training)
        model.alpha -= gap

    model.word2id = dict((w, v.index) for w, v in model.vocab.iteritems())
    model.word_vectors = model.syn0norm

    return model


def run_optimize(cv, text, items, parameters):
    i = 0
    n = np.max(items)
    results = np.zeros([n, 4])
    for train_valid in xrange(n + 1):
        if cv != train_valid:
            training = [text[j] for j, val in enumerate(items) if
                        val != train_valid and val != cv]
            testing = [text[j] for j, val in enumerate(items) if
                       val == train_valid and val != cv]
            ii = 0
            for hs in xrange(2):
                if hs:
                    neg = 0
                else:
                    neg = 20
                for sg in xrange(2):
                    model = run_w2v(text, training, parameters[ii][1], parameters[ii][2], parameters[ii][3],
                                    hs, sg, neg, parameters[ii][0])

                    results[i, ii] = score(model, testing, training, neg, parameters[ii][0])
                    ii += 1
            i += 1
    return np.mean(results, axis=0)


def run_test(cv, text, items, parameters):
    results = np.zeros(4)
    training = [text[j] for j, val in enumerate(items) if
                val != cv]
    testing = [text[j] for j, val in enumerate(items) if
               val == cv]
    ii = 0
    for hs in xrange(2):
        if hs:
            neg = 0
        else:
            neg = 20
        for sg in xrange(2):
            model = run_w2v(text, training, parameters[ii][1], parameters[ii][2], parameters[ii][3],
                            hs, sg, neg, parameters[ii][0])

            results[ii] = score(model, testing, training, neg, parameters[ii][0])
            ii += 1
    return results


def main():
    iteration_arr = xrange(50, 401, 50)
    dim_arr = xrange(50, 501, 50)
    window_arr = xrange(1, 15, 1)
    k = 1500

    items = []
    with open('./Data/cross_validation_w2v.txt', 'r') as f:
        for line in f:
            tmp = line.strip('\n')
            items.append(int(tmp))

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

    print len(items)
    print len(text)
    n = np.max(items) + 1
    print n
    results_cv = np.zeros([n, 4])
    for cv in xrange(n):
        parameters = np.array([[k, 10, 100, 3], [k, 10, 100, 3], [k, 10, 100, 3], [k, 10, 100, 3]])

        for i in xrange(3):
            # print 'CV #%d optimize #%d' % (cv, i)
            # print 'CV #%d iteration' % cv
            results = np.zeros([len(iteration_arr), 4])
            for ii, iteration in enumerate(iteration_arr):
                print 'CV #%d iteration %d of %d' % (cv, ii, len(iteration_arr))
                parameters[:, 2] = iteration
                results[ii, :] = run_optimize(cv, text, items, parameters)

            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmin(results, axis=0)
            parameters[0, 2] = iteration_arr[tmp[0]]
            parameters[1, 2] = iteration_arr[tmp[1]]
            parameters[2, 2] = iteration_arr[tmp[2]]
            parameters[3, 2] = iteration_arr[tmp[3]]

            # print 'CV #%d dim' % cv
            results = np.zeros([len(dim_arr), 4])
            for ii, dim in enumerate(dim_arr):
                print 'CV #%d dim %d of %d' % (cv, ii, len(dim_arr))
                parameters[:, 1] = dim
                results[ii, :] = run_optimize(cv, text, items, parameters)

            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmin(results, axis=0)
            parameters[0, 1] = dim_arr[tmp[0]]
            parameters[1, 1] = dim_arr[tmp[1]]
            parameters[2, 1] = dim_arr[tmp[2]]
            parameters[3, 1] = dim_arr[tmp[3]]

            # print 'CV #%d window' % cv
            results = np.zeros([len(window_arr), 4])
            for ii, window in enumerate(window_arr):
                print 'CV #%d window %d of %d' % (cv, ii, len(window_arr))
                parameters[:, 3] = window
                results[ii, :] = run_optimize(cv, text, items, parameters)

            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmin(results, axis=0)
            parameters[0, 3] = window_arr[tmp[0]]
            parameters[1, 3] = window_arr[tmp[1]]
            parameters[2, 3] = window_arr[tmp[2]]
            parameters[3, 3] = window_arr[tmp[3]]

        print 'CV #%d test' % cv
        results_cv[cv, :] = run_test(cv, text, items, parameters)
        np.save('./Data/Results_w2v/parameters_cv%d.npy' % cv, parameters)
        print results_cv[cv, :]
        print parameters
    np.save('./Data/Results_w2v/results.npy', results_cv)


if __name__ == '__main__':
    main()
