import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import kurtosis, skew
import sklearn.model_selection
from MetaFeatures.AbstractFeature import AbstractFeature


class LandmarkLda(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        import sklearn.discriminant_analysis
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFolf(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                lda.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            else:
                lda = OneVsRestClassifier(lda)
                lda.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            predictions = lda.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y.iloc[test])
        return accuracy / 10


class LandmarkNaiveBayes(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        import sklearn.naive_bayes
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            nb = sklearn.naive_bayes.GaussianNB()

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                nb.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            else:
                nb = OneVsRestClassifier(nb)
                nb.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            predictions = nb.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y.iloc[test])
        return accuracy / 10


class LandmarkDeicisionTree(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
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
                tree.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            predictions = tree.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y.iloc[test])
        return accuracy / 10


class LandmarkDecisionNodeLearner(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
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
                node.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            else:
                node = OneVsRestClassifier(node)
                node.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            predictions = node.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y.iloc[test])
        return accuracy / 10


class LankmarkRandomNodeLearner(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
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
            node.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            predictions = node.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y.iloc[test])
        return accuracy / 10


class Landmark1NN(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        import sklearn.neighbors
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.
        for train, test in kf.split(self.X, self.y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kNN.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(self.X.iloc[train], np.ravel(self.y.iloc[train],order='C'))
            predictions = kNN.predict(self.X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y.iloc[test])
        return accuracy / 10


class PCA(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(self.X.shape[0])
        for i in range(10):
            rs.shuffle(indices)
            pca.fit(self.X.iloc[indices])
            return pca


class PCA95Percent(AbstractFeature):

    def __init__(self, X, y, *args):
        self.pca = args[0]
        super().__init__(X, y)

    def calculate(self):
        sum = 0
        idx = 0
        while sum < 0.95 and idx < len(self.pca.explained_variance_ratio_):
            sum += self.pca.explained_variance_ratio_[idx]
            idx += 1
        return float(idx) / float(self.X.shape[1])


class PCAKurtosisFirstPc(AbstractFeature):

    def __init__(self, X, y, *args):
        self.pca = args[0]
        super().__init__(X, y)

    def calculate(self):
        components = self.pca.components_
        self.pca.components_ = components[:1]
        transformed = self.pca.transform(self.X)
        self.pca.components_ = components
        temp = kurtosis(transformed)
        return temp[0]


class PCASkewnessFirstPc(AbstractFeature):

    def __init__(self, X, y, *args):
        self.pca = args[0]
        super().__init__(X, y)

    def calculate(self):
        components = self.pca.components_
        self.pca.components_ = components[:1]
        transformed = self.pca.transform(self.X)
        self.pca.components_ = components
        temp = skew(transformed)
        return temp[0]


class ClusteringFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
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
        super().__init__(X, y)

    def calculate(self):
        self.landmark_1NN = Landmark1NN(self.X, self.y).value
        self.landmark_decision_node_learner = LandmarkDecisionNodeLearner(self.X, self.y).value
        self.landmark_decision_tree = LandmarkDeicisionTree(self.X, self.y).value
        self.landmark_lda = LandmarkLda(self.X, self.y).value
        self.landmark_naive_bayes = LandmarkNaiveBayes(self.X, self.y).value
        self.landmark_random_node_learner = LankmarkRandomNodeLearner(self.X, self.y).value
        self.pca = PCA(self.X, self.y).value
        self.pca_95percent = PCA95Percent(self.X, self.y, self.pca).value
        self.pca_kurtosis_first_pc = PCAKurtosisFirstPc(self.X, self.y, self.pca).value
        self.pca_skewness_first_pc = PCASkewnessFirstPc(self.X, self.y, self.pca).value
        return [self.landmark_1NN, self.landmark_decision_node_learner,
                self.landmark_decision_tree, self.landmark_lda,
                self.landmark_naive_bayes, self.landmark_random_node_learner,
                self.pca_95percent, self.pca_kurtosis_first_pc,
                self.pca_skewness_first_pc]
