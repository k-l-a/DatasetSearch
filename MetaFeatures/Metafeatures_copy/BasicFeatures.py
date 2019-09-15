import MetaFeatures.AbstractFeature.AbstractFeature
import math


class BasicFeatures(AbstractFeature):

    def __init__(self, X, y):
        super().__init__(X, y)
        self.log_number_of_features = None
        self.log_number_of_instances = None
        self.number_of_features = None
        self.number_of_instances = None
        self.calculate()

    # number-of-instances
    def get_number_of_instances(self):
        self.number_of_instances = len(self.y)
        return self.number_of_instances

    # log-number-of-instances
    def get_log_number_of_instances(self):
        if self.number_of_instances is None:
            self.get_number_of_instances()
        self.log_number_of_instances = math.log(self.number_of_instances)
        return self.log_number_of_instances

    # number-of-features
    def get_number_of_features(self):
        self.number_of_features = len(self.X.columns)
        return self.number_of_features

    # log-number-of-features
    def get_log_number_of_features(self):
        if self.number_of_features is None:
            self.get_number_of_features()
        self.log_number_of_features = math.log(self.number_of_features)
        return self.log_number_of_features

    def calculate(self):
        self.get_number_of_features()
        self.get_number_of_instances()
        self.get_log_number_of_features()
        self.get_log_number_of_instances()
        return list([self.log_number_of_features, self.log_number_of_instances,
                     self.number_of_features, self.number_of_instances])
