import numpy as np
import math
from scipy.stats import entropy
from MetaFeatures.AbstractFeature import AbstractFeature


class NumberOfClasses(AbstractFeature):

    def __init__(self, X, y, *args):
        self.classes = None
        self.class_occurrence = None
        super().__init__(X, y)

    def calculate(self):
        self.classes, self.class_occurrence = np.unique(self.y, return_counts=True)
        return len(self.classes)


class ClassProbability(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_occurrence = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.class_occurrence / len(self.y)


class ClassProbabilityMax(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_probability = args[0]
        super().__init__(X, y)

    def calculate(self):
        if len(self.class_probability) == 1:
            return 1
        return self.class_probability.max()


class ClassProbabilityMean(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_probability = args[0]
        super().__init__(X, y)

    def calculate(self):
        if len(self.class_probability) == 1:
            return 1
        return self.class_probability.mean()


class ClassProbabilityMin(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_probability = args[0]
        super().__init__(X, y)

    def calculate(self):
        if len(self.class_probability) == 1:
            return 0
        return self.class_probability.min()


class ClassProbabilityStd(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_probability = args[0]
        super().__init__(X, y)

    def calculate(self):
        if len(self.class_probability) == 1:
            return 0
        return self.class_probability.std()


class DatasetRatio(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_probability_max = args[0]
        super().__init__(X, y)

    def calculate(self):
        if self.class_probability_max == 1:
            return len(self.y)
        return 1 / (1 - self.class_probability_max) - 1


class LogDatasetRatio(AbstractFeature):

    def __init__(self, X, y, *args):
        self.dataset_ratio = args[0]
        super().__init__(X, y)

    def calculate(self):
        return math.log(self.dataset_ratio)


class InverseDatasetRatio(AbstractFeature):

    def __init__(self, X, y, *args):
        self.dataset_ratio = args[0]
        super().__init__(X, y)

    def calculate(self):
        return 1 / self.dataset_ratio


class LogInverseDatasetRatio(AbstractFeature):

    def __init__(self, X, y, *args):
        self.inverse_dataset_ratio = args[0]
        super().__init__(X, y)

    def calculate(self):
        return math.log(self.inverse_dataset_ratio)


class ClassEntropy(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_probability = args[0]
        super().__init__(X, y)

    def calculate(self):
        return entropy(self.class_probability)


class ClassFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        self.class_entropy = None
        self.class_probability_max = None
        self.class_probability_mean = None
        self.class_probability_min = None
        self.class_probability_std = None
        self.dataset_ratio = None
        self.inverse_dataset_ratio = None
        self.log_dataset_ratio = None
        self.log_inverse_dataset_ratio = None
        self.number_of_classes = None
        self.class_probability = None
        super().__init__(X, y)

    def calculate(self):
        self.number_of_classes = NumberOfClasses(self.X, self.y)
        self.class_probability = ClassProbability(self.X, self.y, self.number_of_classes.class_occurrence).value
        self.number_of_classes = self.number_of_classes.value
        self.class_probability_max = ClassProbabilityMax(self.X, self.y, self.class_probability).value
        self.class_probability_mean = ClassProbabilityMean(self.X, self.y, self.class_probability).value
        self.class_probability_min = ClassProbabilityMin(self.X, self.y, self.class_probability).value
        self.class_probability_std = ClassProbabilityStd(self.X, self.y, self.class_probability).value
        self.dataset_ratio = DatasetRatio(self.X, self.y, self.class_probability_max).value
        self.inverse_dataset_ratio = InverseDatasetRatio(self.X, self.y, self.dataset_ratio).value
        self.log_dataset_ratio = LogDatasetRatio(self.X, self.y, self.dataset_ratio).value
        self.log_inverse_dataset_ratio = LogInverseDatasetRatio(self.X, self.y, self.inverse_dataset_ratio).value
        self.class_entropy = ClassEntropy(self.X, self.y, self.class_probability).value
        return [self.class_entropy, self.class_probability_max, self.class_probability_mean,
                     self.class_probability_min, self.class_probability_std, self.dataset_ratio,
                     self.inverse_dataset_ratio, self.log_dataset_ratio, self.log_inverse_dataset_ratio,
                     self.number_of_classes]
