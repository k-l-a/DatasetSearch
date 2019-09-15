import numpy as np

class MissingValuesFeatures:

    def __init__(self, basic_features):
        self.basic_features = basic_features
        self.X = basic_features.X
        self.y = basic_features.y
        self.number_of_instances_with_missing_values = None
        self.number_of_features_with_missing_values = None
        self.number_of_missing_values = None
        self.percentage_of_instances_with_missing_values = None
        self.percentage_of_features_with_missing_values = None
        self.percentage_of_missing_values = None

    # number-of-instances-with-missing-values
    def get_number_of_instances_with_missing_values(self):
        temp = np.arange(self.basic_features.number_of_instances)
        f = lambda x: np.isnan(x).any()
        g = lambda x: self.X.loc[x]
        self.number_of_instances_with_missing_values = sum(f(g(temp)))
        return self.number_of_instances_with_missing_values

    # percentage-of-Instances-with-missing-values
    def get_percentage_of_instances_with_missing_values(self):
        if self.number_of_instances_with_missing_values is None:
            self.get_number_of_instances_with_missing_values()
        self.percentage_of_instances_with_missing_values = \
            self.number_of_instances_with_missing_values / self.basic_features.number_of_instances
        return self.percentage_of_instances_with_missing_values

    # number-of-features-with-missing-values
    def get_number_of_features_with_missing_values(self):
        temp = np.arange(self.basic_features.number_of_features)
        f = lambda x: np.isnan(x).any()
        g = lambda x: self.X[x]
        temp = f(g(temp)).sum()
        self.number_of_features_with_missing_values = temp
        return self.number_of_features_with_missing_values

    # percentage-of-features-with-missing-values
    def get_percentage_of_features_with_missing_values(self):
        if self.number_of_features_with_missing_values is None:
            self.get_number_of_features_with_missing_values()
        self.percentage_of_features_with_missing_values = \
            self.number_of_features_with_missing_values / self.basic_features.number_of_features
        return self.percentage_of_features_with_missing_values

    # number-of-missing-values
    def get_number_of_missing_values(self):
        self.number_of_missing_values = self.X.isnull().sum().sum()
        return self.number_of_missing_values

    # percentage-of-missing-values
    def get_percentage_of_missing_values(self):
        if self.number_of_missing_values is None:
            self.get_number_of_missing_values()
        self.percentage_of_missing_values = \
            self.number_of_missing_values / self.basic_features.number_of_features / self.basic_features.number_of_instances

    def calculate(self):
        self.get_number_of_instances_with_missing_values()
        self.get_number_of_features_with_missing_values()
        self.get_number_of_missing_values()
        self.get_percentage_of_missing_values()
        self.get_percentage_of_instances_with_missing_values()
        self.get_percentage_of_features_with_missing_values()
        return list([self.number_of_instances_with_missing_values,
                     self.number_of_features_with_missing_values,
                     self.number_of_missing_values,
                     self.percentage_of_instances_with_missing_values,
                     self.get_percentage_of_features_with_missing_values(),
                     self.get_percentage_of_missing_values()])
