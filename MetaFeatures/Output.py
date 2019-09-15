from MetaFeatures.BasicFeatures import BasicFeatures
from MetaFeatures.ClusteringFeatures import ClusteringFeatures
from MetaFeatures.DatatypeFeatures import CategoricalFeatures
from MetaFeatures.ClassFeatures import ClassFeatures
from MetaFeatures.DistributionFeatures import DistributionFeatures
from MetaFeatures.MissingDataFeatures import MissingValuesFeatures
from MetaFeatures.AddedFeatures import AdFeatures
from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np


class MetaFeatures:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def calculate(self):
        list1 = self.get_basic_features()
        list2 = self.get_class_features()
        list3 = self.get_distribution_features()
        list4 = self.get_categorical_features()
        list5 = self.get_missing_values_features()
        list6 = self.get_clustering_features()
        list7 = self.get_adfeatures()
        raw = list1 + list2 + list3 + list4 + list5 + list6 + list7
        # norm = [float(i) / max(raw) for i in raw]
        return raw

    def get_basic_features(self):
        return BasicFeatures(self.X, self.y).value

    def get_clustering_features(self):
        return ClusteringFeatures(self.X, self.y).value

    def get_categorical_features(self):
        return CategoricalFeatures(self.X, self.y).value

    def get_class_features(self):
        return ClassFeatures(self.X, self.y).value

    def get_distribution_features(self):
        return DistributionFeatures(self.X, self.y).value

    def get_missing_values_features(self):
        return MissingValuesFeatures(self.X, self.y).value
    
    def get_adfeatures(self):
        return AdFeatures(self.X, self.y).value


def get_metafeatures(path, type='raw'):
    X_train, y_train = load_svmlight_file(path)
    mat = X_train.todense()
    X = pd.DataFrame(mat)
    X.columns = range(len(X.columns))
    y = pd.DataFrame(y_train)
    y.columns = ['target']
    metafeatures = MetaFeatures(X, y).calculate()
    f = lambda x: x.item() if isinstance(x, np.generic) else x
    metafeatures = [f(i) for i in metafeatures]
    if type == "norm":
        return [float(i) / max(metafeatures) for i in metafeatures]
    return metafeatures
