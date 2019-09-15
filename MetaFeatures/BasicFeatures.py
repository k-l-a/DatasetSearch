from MetaFeatures.AbstractFeature import AbstractFeature
import math


class NumberOfFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        return len(self.X.columns)


class LogNumberOfFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        if isinstance(args[0], int):
            self.number_of_features = args[0]
        super().__init__(X, y)

    def calculate(self):
        return math.log(self.number_of_features)


class NumberOfInstances(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        return len(self.y)


class LogNumberOfInstances(AbstractFeature):

    def __init__(self, X, y, *args):
        if isinstance(args[0], int):
            self.number_of_instances = args[0]
        super().__init__(X, y)

    def calculate(self):
        return math.log(self.number_of_instances)


class BasicFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        self.log_number_of_features = None
        self.log_number_of_instances = None
        self.number_of_features = None
        self.number_of_instances = None
        super().__init__(X, y)

    def calculate(self):
        self.number_of_features = NumberOfFeatures(self.X, self.y).value
        self.number_of_instances = NumberOfInstances(self.X, self.y).value
        self.log_number_of_features = LogNumberOfFeatures(self.X, self.y, self.number_of_features).value
        self.log_number_of_instances = LogNumberOfInstances(self.X, self.y, self.number_of_instances).value
        return [self.number_of_features, self.number_of_instances,
                     self.log_number_of_features, self.log_number_of_instances]
