from MetaFeatures.BasicFeatures import BasicFeatures
from MetaFeatures.ClusteringFeatures import ClusteringFeatures
from MetaFeatures.DatatypeFeatures import CategoricalFeatures
from MetaFeatures.ClassFeatures import ClassFeatures
from MetaFeatures.DistributionFeatures import DistributionFeatures
from MetaFeatures.MissingDataFeatures import MissingValuesFeatures


class MetaFeatures:

    def __init__(self, X, y):
        self.basic_features = BasicFeatures(X, y)
        self.clustering_features = ClusteringFeatures(self.basic_features)
        self.categorical_features = CategoricalFeatures(self.basic_features)
        self.class_features = ClassFeatures(self.basic_features)
        self.distribution_features = DistributionFeatures(self.basic_features)
        self.missing_values_features = MissingValuesFeatures(self.basic_features)

    def calculate(self):
        list1 = self.basic_features.calculate()
        list6 = self.clustering_features.calculate()
        list4 = self.categorical_features.calculate()
        list2 = self.class_features.calculate()
        list3 = self.distribution_features.calculate()
        list5 = self.missing_values_features.calculate()
        return list1 + list2 + list3 + list4 + list5 + list6
