import numpy as np
from MetaFeatures.AbstractFeature import AbstractFeature


class NumberOfInstancesWithMissingValues(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        temp = np.arange(len(self.y))
        f = lambda x: np.isnan(x).any()
        g = lambda x: self.X.reindex([x])
        return sum(f(g(temp)))


class PercentageOfInstancesWithMissingValues(AbstractFeature):

    def __init__(self, X, y, *args):
        self.number_of_instances_with_missing_values = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.number_of_instances_with_missing_values / len(self.y)


class NumberOFFeaturesWithMissingValues(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: np.isnan(x).any()
        g = lambda x: self.X.iloc[:, x]
        temp = f(g(temp)).sum()
        return temp


class PercentageOfFeaturesWithMissingValues(AbstractFeature):

    def __init__(self, X, y, *args):
        self.number_of_features_with_missing_values = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.number_of_features_with_missing_values / len(self.X.columns)


class NumberOfMissingValues(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        return self.X.isnull().sum().sum()


class PercentageOfMissingValues(AbstractFeature):

    def __init__(self, X, y, *args):
        self.number_of_missing_values = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.number_of_missing_values / len(self.X.columns) / len(self.y)


class MissingValuesFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        self.number_of_instances_with_missing_values = None
        self.number_of_features_with_missing_values = None
        self.number_of_missing_values = None
        self.percentage_of_instances_with_missing_values = None
        self.percentage_of_features_with_missing_values = None
        self.percentage_of_missing_values = None
        super().__init__(X, y)

    def calculate(self):
        self.number_of_instances_with_missing_values = NumberOfInstancesWithMissingValues(self.X, self.y).value
        self.number_of_features_with_missing_values = NumberOFFeaturesWithMissingValues(self.X, self.y).value
        self.number_of_missing_values = NumberOfMissingValues(self.X, self.y).value
        self.percentage_of_instances_with_missing_values = PercentageOfInstancesWithMissingValues(self.X, self.y, self.number_of_instances_with_missing_values).value
        self.percentage_of_features_with_missing_values = PercentageOfFeaturesWithMissingValues(self.X, self.y, self.number_of_features_with_missing_values).value
        self.percentage_of_missing_values = PercentageOfMissingValues(self.X, self.y, self.number_of_missing_values).value
        return [self.number_of_instances_with_missing_values,
                     self.number_of_features_with_missing_values,
                     self.number_of_missing_values,
                     self.percentage_of_instances_with_missing_values,
                     self.percentage_of_features_with_missing_values,
                     self.percentage_of_missing_values]
