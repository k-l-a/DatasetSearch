import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import kurtosis, skew
import sklearn.model_selection


class ClusteringFeatures:

    def __init__(self, basic_features):
        self.basic_features = basic_features
        self.X = basic_features.X
        self.y = basic_features.y
        self.landmark_1NN = None
        self.landmark_decision_node_learner = None
        self.landmark_decision_tree = None
        self.landmark_lda = None
        self.landmark_naive_bayes = None
        self.landmark_random_node_learner = None
        self.pca_95percent = None
        self.pca_kurtosis_first_pc = None
        self.pca_skewness_first_pc = None
        self.pca = None

    def get_landmark_lda(self):
        import sklearn.discriminant_analysis
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFolf(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                lda.fit(self.X.iloc[train], self.y.iloc[train])
            else:
                lda = OneVsRestClassifier(lda)
                lda.fit(self.X.iloc[train], self.y.iloc[train])
            predictions = lda.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        self.landmark_lda = accuracy / 10
        return self.landmark_lda

    def get_landmark_naive_bayes(self):
        import sklearn.naive_bayes
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            nb = sklearn.naive_bayes.GaussianNB()

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                nb.fit(self.X.iloc[train], self.y.iloc[train])
            else:
                nb = OneVsRestClassifier(nb)
                nb.fit(self.X.iloc[train], self.y.iloc[train])
            predictions = nb.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        self.landmark_naive_bayes = accuracy / 10
        return self.landmark_naive_bayes

    def get_landmark_decision_tree(self):
        import sklearn.tree
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            random_state = sklearn.utils.check_random_state(42)
            tree = sklearn.tree.DecisionTreeClassifier(random_state=random_state)
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                tree.fit(self.X.iloc[train], self.y.iloc[train])
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(self.X.iloc[train], self.y.iloc[train])
            predictions = tree.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        self.landmark_decision_tree = accuracy / 10
        return self.landmark_decision_tree

    def get_landmark_decision_node_learner(self):
        import sklearn.tree
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=None)
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                node.fit(self.X.iloc[train], self.y.iloc[train])
            else:
                node = OneVsRestClassifier(node)
                node.fit(self.X.iloc[train], self.y.iloc[train])
            predictions = node.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        self.landmark_decision_node_learner = accuracy / 10
        return self.landmark_decision_node_learner

    def get_landmark_random_node_learner(self):
        import sklearn.tree
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=1)
            node.fit(self.X.iloc[train], self.y.iloc[train])
            predictions = node.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        self.landmark_random_node_learner = accuracy / 10
        return self.landmark_random_node_learner

    def get_landmark_1NN(self):
        import sklearn.neighbors
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kNN.fit(self.X.iloc[train], self.y.iloc[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(self.X.iloc[train], self.y.iloc[train])
            predictions = kNN.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        self.landmark_1NN = accuracy / 10
        return self.landmark_1NN

    def get_pca(self):
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(self.X.shape[0])
        for i in range(10):
            rs.shuffle(indices)
            pca.fit(self.X.iloc[indices])
            self.pca = pca
            return self.pca

    def get_pca_95percent(self):
        if self.pca is None:
            self.get_pca()
        sum = 0
        idx = 0
        while sum < 0.95 and idx < len(self.pca.explained_variance_ratio_):
            sum += self.pca.explained_variance_ratio_[idx]
            idx += 1
        self.pca_95percent = float(idx) / float(self.X.shape[1])
        return self.pca_95percent

    def get_pca_kurtosis_first_pc(self):
        if self.pca is None:
            self.get_pca()
        components = self.pca.components_
        self.pca.components_ = components[:1]
        transformed = self.pca.transform(self.X)
        self.pca.components_ = components
        temp = kurtosis(transformed)
        self.pca_kurtosis_first_pc = temp[0]
        return self.pca_kurtosis_first_pc

    def get_pca_skewness_first_pc(self):
        if self.pca is None:
            self.get_pca()
        components = self.pca.components_
        self.pca.components_ = components[:1]
        transformed = self.pca.transform(self.X)
        self.pca.components_ = components
        temp = skew(transformed)
        self.pca_skewness_first_pc = temp[0]
        return self.pca_skewness_first_pc

    def calculate(self):
        self.get_landmark_1NN()
        self.get_landmark_decision_node_learner()
        self.get_landmark_decision_tree()
        self.get_landmark_lda()
        self.get_landmark_naive_bayes()
        self.get_landmark_random_node_learner()
        self.get_pca()
        self.get_pca_95percent()
        self.get_pca_kurtosis_first_pc()
        self.get_pca_skewness_first_pc()
        return list([self.landmark_1NN, self.landmark_decision_node_learner,
                     self.landmark_decision_tree, self.landmark_lda,
                     self.landmark_naive_bayes, self.landmark_random_node_learner,
                     self.pca_95percent, self.pca_kurtosis_first_pc,
                     self.pca_skewness_first_pc])
