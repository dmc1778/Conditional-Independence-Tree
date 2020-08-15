import ctypes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def code(x_slow, y_slow):
    temp_x = []
    temp_y = []
    compact_attribute = []
    for i in range(len(x_slow[0])):
        temp_x = temp_x + [[]]
        compact_attribute = compact_attribute + [list(set([x_slow[j][i] for j in range(len(x_slow))]))]
        a = [compact_attribute[i].index(x_slow[j][i]) for j in range(len(x_slow))]
        for j in range(len(x_slow)):
            temp_x[i] = temp_x[i] + [a[j]]
    compact_attribute = compact_attribute + [list(set([y_slow[j] for j in range(len(x_slow))]))]
    a = [compact_attribute[len(x_slow[0])].index(y_slow[j]) for j in range(len(x_slow))]
    for j in range(len(x_slow)):
        temp_y = temp_y + [a[j]]

    x = (ctypes.POINTER(ctypes.c_int) * len(x_slow))()
    y = (ctypes.c_int * len(x_slow))()
    for j in range(len(x_slow)):
        x[j] = (ctypes.c_int * len(x_slow[0]))()
        for i in range(len(x_slow[0])):
            x[j][i] = temp_x[i][j]
        y[j] = temp_y[j]
    return x, y, compact_attribute


class BuildStructure:
    def __init__(self, x_slow, y_slow):
        self.I = []
        self.h = []
        self.h_states = []
        x, y, compact_attribute = code(x_slow, y_slow)
        x_dim = len(y_slow)
        y_dim = len(x_slow[0])
        mutual = (ctypes.POINTER(ctypes.c_double) * y_dim)()
        for i in range(y_dim):
            mutual[i] = (ctypes.c_double * y_dim)()
        x_dim = ctypes.c_int(x_dim)
        y_dim = ctypes.c_int(y_dim)

        imc = ctypes.CDLL('E:\\apply\\york\\project\\source\\find_mutual_conditional.dll')
        imc.information_mutual_conditional_all(x, y, x_dim, y_dim, mutual)

        y_dim = y_dim.value
        for i in range(y_dim):
            self.I = self.I + [[]]
            for j in range(y_dim):
                self.I[i] = self.I[i] + [mutual[i][j]]

    def structure(self, n_h):
        y_dim = len(self.I)
        cluster = []
        for i in range(y_dim):
            # initial cluster size which is equal to the number of attributes in the dataset
            cluster = cluster + [set([i])]
        # n_h is the number of clusters determined by the user
        while len(cluster) > n_h:
            distance = []
            for i in range(len(cluster)):
                distance = distance + [[]]
                for _ in range(len(cluster)):
                    distance[i] = distance[i] + [0]
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    distance[i][j] = 0
                    for k in cluster[i]:
                        for l in cluster[j]:
                            distance[i][j] += self.I[k][l]
                    distance[i][j] /= (len(cluster[i]) * len(cluster[j]))
                    distance[j][i] = distance[i][j]
            index_i = 0
            index_j = 1
            maximum = distance[0][1]
            for i in range(len(distance)):
                for j in range(i + 1, len(distance)):
                    if maximum < distance[i][j]:
                        maximum = distance[i][j]
                        index_i = i
                        index_j = j
            cluster[index_i] = cluster[index_i] | cluster[index_j]
            del cluster[index_j]

        tree = []
        for i in range(len(cluster)):
            if len(cluster[i]) > 1:
                tree.append(list(cluster[i]))

        return tree


class ConditionalIndependenceTree:
    def __init__(self, alpha):
        self.alpha = alpha

    def make_prob(self, data, tree_indicator):
        tree_params = []
        uc = np.unique(data.iloc[:, -1])

        for i in range(len(tree_indicator)):
            tree_params = tree_params + [[]]
            current_tree = tree_indicator[i]
            for c in range(len(uc)):
                tree_params[i] = tree_params[i] + [[]]
                for j in range(len(current_tree)):
                    aj = current_tree[j]
                    uai = np.unique(data.iloc[:, aj])
                    tree_params[i][c] = tree_params[i][c] + [[]]
                    for k in range(np.size(uai, 0)):
                        tree_params[i][c][j] = tree_params[i][c][j] + [[]]

        return tree_params, tree_indicator

    def fit(self, X_train, y_train):
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        data = pd.concat((X_train, y_train), axis=1)

        self.clust = self.identify_subtree(data)

        self.tree_params, self.tree_indicator = self.make_prob(data, self.clust)

        n, N = data.shape
        uc = np.unique(data.iloc[:, -1])
        self.pclass = np.zeros((1, len(uc)))
        for c in range(len(uc)):
            self.pclass[0, c] = len(data[data.iloc[:, -1] == uc[c]]) / n

        self.unique_holder = []
        self.size_holder = []
        for i in range(len(self.tree_params)):
            self.unique_holder = self.unique_holder + [[]]
            self.size_holder = self.size_holder + [[]]
            current_cluster = self.tree_indicator[i]
            for j in range(len(current_cluster)):
                aj = current_cluster[j]
                uaj = list(np.unique(data.iloc[:, aj]))
                self.unique_holder[i] = self.unique_holder[i] + [uaj]
                self.size_holder[i] = self.size_holder[i] + [len(uaj)]

        for i in range(len(self.tree_params)):
            current_cluster = self.tree_indicator[i]
            for c in range(len(uc)):
                for j in range(len(current_cluster)):
                    aj = current_cluster[j]
                    for k in range(self.size_holder[i][j]):
                        idx = self.unique_holder[i][j]
                        self.tree_params[i][c][j][k] = (
                                len(np.where((data.iloc[:, aj] == idx[k]) & (data.iloc[:, -1] == uc[c]))[0]) / len(
                            np.where((data.iloc[:, -1] == uc[c]))[0]))
                        if self.tree_params[i][c][j][k] == 0:
                            self.tree_params[i][c][j][k] = self.alpha
                        self.tree_params[i][c][j][k] = np.log2(self.tree_params[i][c][j][k])
        return self

    def predict(self, test_data):
        test_data = pd.DataFrame(test_data)
        pred = []
        for t in range(len(test_data)):
            instance = test_data.iloc[t, :]
            output = []
            prob = []
            for c in range(np.size(self.pclass, 1)):
                ti = 0
                prob = prob + [[]]
                for i in range(np.size(self.tree_params, 0)):
                    current_cluster = self.tree_indicator[i]
                    temp = 0
                    for j in range(np.size(current_cluster, 0)):
                        aj = current_cluster[j]
                        idx = instance.iloc[aj]
                        if idx in self.unique_holder[i][j]:
                            index = self.unique_holder[i][j].index(idx)
                            temp += self.tree_params[i][c][j][index]
                        else:
                            temp += np.log2(1 / np.size(self.pclass, 1))
                    ti += temp
                prob[c] = ti
                prob[c] = 2 ** prob[c]
            output.append(prob)
            pred.append(np.argmax(prob)+1)
        return pred

    def identify_subtree(self, data):
        X = data.iloc[:, 0:-1].values
        y = data.iloc[:, -1].values
        struct = BuildStructure(X, y)
        n, N = data.shape
        if N - 1 <= 5:
            n_cluster = 2
        else:
            n_cluster = round(np.divide(N - 1, 2))
        sub_trees = struct.structure(n_cluster)
        return sub_trees


def main():
    citree = ConditionalIndependenceTree(alpha=0.01)

    citree.fit(train_data)
    pred = citree.predict(test_data)

    print("CITree Classification Accuracy:", accuracy_score(test_data.iloc[:, -1], pred))
    print(classification_report(test_data.iloc[:, -1], pred))
    print("AUC:", roc_auc_score(test_data.iloc[:, -1], pred, average=None))


if __name__ == '__main__':
    main()
