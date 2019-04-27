import numpy as np
import cvxopt
import pickle as pkl
import os
import matplotlib.pyplot as plt


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        # prevent solver verbose
        cvxopt.solvers.options['show_progress'] = False
        # initialize
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


def soft_classifier(c):
    # get the feature list
    feature_list = os.listdir('../../Data/Pickle/flatten_features/')
    feature_list.sort(key=lambda x: int(x.split('_')[2]))
    feature_list = np.array(feature_list)

    # random split train and test
    indices = np.arange(len(feature_list))
    np.random.shuffle(indices)
    train_ind, test_ind = indices[:int(0.6*len(feature_list))], indices[int(0.6*len(feature_list)):]

    # split features in training and testing
    train_features = feature_list[train_ind]
    test_features = feature_list[test_ind]
    print('testfeat cnt', len(test_features))

    # get the labels
    labels = pkl.load(open('../../Data/Pickle/feat_label_map', 'rb'))
    labels = np.array([labels[key] for key in labels.keys()])
    temp_label = []
    for i in range(feature_list.shape[0]):
        if i == feature_list.shape[0]:
            t = labels[:-90, :]
        t = labels[128*i:128*(i+1), :]
        temp_label.append(t)
    labels = np.array(temp_label)

    # split labels in training and testing
    train_labels = []
    for i in train_ind:
        train_labels.append(labels[i])
    train_labels = np.array(train_labels)
    test_labels = []
    for i in test_ind:
        test_labels.append(labels[i])
    test_labels = np.array(test_labels)

    # list for correct predictions
    pred_cnt = []

    # for each label
    for label_ind in range(9):
        print("Classfier for label", label_ind)
        # define the classifier with C value
        clf = SVM(C=c)
        cnt = 0

        # train the classifier
        for i in range(train_features.shape[0]):
            print("Training batch", cnt + 1)
            data = pkl.load(open('../../Data/Pickle/flatten_features/' + train_features[i], 'rb'))
            labels_batch = train_labels[i][:, label_ind]
            clf.fit(data, labels_batch)
            cnt += 1

        # testing parameters
        correct = 0
        total = 0
        cnt = 0

        # test the classifier
        for i in range(test_features.shape[0]):
            print("Testing batch", cnt + 1)
            test_data = pkl.load(open('../../Data/Pickle/flatten_features/'+test_features[i], 'rb'))
            label_test = test_labels[i][:, label_ind]
            y_predict = clf.predict(test_data)
            total += len(y_predict)
            correct += np.sum(y_predict == label_test)
            print("correct", correct)
            cnt += 1

        print("%d out of %d predictions correct" % (correct, total))
        pred_cnt.append(correct)

    print('pred_cnt',pred_cnt)
    pkl.dump(pred_cnt, open('../../Data/Pickle/svm/pred_cnt_'+str(c)+'.p', 'wb'))
    plot(pred_cnt, total, c)


def plot(pred_cnt, total, C):
    y_pos = range(9)
    labels = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive', 'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
    pred_cnt = np.array(pred_cnt)/total
    plt.bar(y_pos, pred_cnt, align='center')
    plt.xticks(y_pos, labels)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('Accuracy')
    plt.title('SVM with C='+str(C))
    plt.savefig('../../Data/Plots/svm_bar_'+str(C)+'.png')
    plt.show()

def plot_all():
    C = [10, 50, 100, 200, 500, 1000]
    labels = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
              'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
    color = ['brown', 'r', 'tomato', 'lightsalmon', 'saddlebrown', 'linen']
    total = 734 * 128
    y_pos = np.arange(9)
    w = 0.09
    p = []
    for i in range(len(C)):
        pred_cnt = np.array(pkl.load(open('../../Data/Pickle/svm/pred_cnt_' + str(C[i]) + '.p', 'rb')))
        pred_cnt = np.array(pred_cnt) / total
        p.append(plt.bar(y_pos + (w * (i + 1)), pred_cnt, width=w, color=color[i]))
    plt.xticks(y_pos + w * 3, labels)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('Accuracy')
    plt.title('Soft Margin SVM')
    plt.legend((p), (C), title='C')
    plt.savefig('../../Data/Plots/svm_bar_'+str(C)+'.png')
    plt.show()

if __name__ == "__main__":
    C = [10, 50, 100, 200, 500, 1000]
    total = 734 * 128
    for c in C:
        print("C:",c)
        soft_classifier(c)
        # pred_cnt = np.array(pkl.load(open('../../Data/Pickle/svm/pred_cnt_'+str(C[i])+'.p', 'rb')))
        # plot(pred_cnt, total, c)

