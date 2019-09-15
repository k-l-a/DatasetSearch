import numpy as np
import math
from scipy.stats import entropy

class ClassFeatures:

    def __init__(self, basic_features):
        self.basic_features = basic_features
        self.X = basic_features.X
        self.y = basic_features.y
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
        self.class_occurrence = None
        self.class_probability = None

    # number-of-classes
    def get_number_of_classes(self):
        classes, self.class_occurrence = np.unique(self.y, return_counts=True)
        self.number_of_classes = len(classes)
        return self.number_of_classes

    # class-probability
    def get_class_probability(self):
        if self.number_of_classes is None:
            self.get_number_of_classes()
        self.class_probability = self.class_occurrence / self.basic_features.number_of_instances
        return self.class_probability

    # class_probability_max
    def get_class_probability_max(self):
        if self.class_probability is None:
            self.get_class_probability()
        self.class_probability_max = self.class_probability.max()
        return self.class_probability_max

    # class-probability-mean
    def get_class_probability_mean(self):
        if self.class_probability is None:
            self.get_class_probability()
        self.class_probability_mean = self.class_probability.mean()
        return self.class_probability_mean

    # class-probability_min
    def get_class_probability_min(self):
        if self.class_probability is None:
            self.get_class_probability()
        self.class_probability_min = self.class_probability.min()
        return self.class_probability_min

    # class-probability-std
    def get_class_probability_std(self):
        if self.class_probability is None:
            self.get_class_probability()
        self.class_probability_std = self.class_probability.std()
        return self.class_probability_std

    # dataset-ratio
    # 1-vs-all
    def get_dataset_ratio(self):
        if self.class_probability_max is None:
            self.get_class_probability_max()
        self.dataset_ratio = 1 / (1 - self.class_probability_max) - 1
        return self.dataset_ratio

    # log-dataset-ratio
    # 1-vs-all
    def get_log_dataset_ratio(self):
        if self.dataset_ratio is None:
            self.get_dataset_ratio()
        self.log_dataset_ratio = math.log(self.dataset_ratio)
        return self.log_dataset_ratio

    # inverse-dataset-ratio
    def get_inverse_dataset_ratio(self):
        if self.dataset_ratio is None:
            self.get_dataset_ratio()
        self.inverse_dataset_ratio = 1 / self.dataset_ratio
        return self.inverse_dataset_ratio

    # log-inverse-dataset-ratio
    def get_log_inverse_dataset_ratio(self):
        if self.inverse_dataset_ratio is None:
            self.get_inverse_dataset_ratio()
        self.log_inverse_dataset_ratio = math.log(self.inverse_dataset_ratio)
        return self.inverse_dataset_ratio

    # class-entropy
    def get_class_entropy(self):
        if self.class_probability is None:
            self.get_class_probability()
        self.class_entropy = entropy(self.class_probability)
        return self.class_entropy

    def calculate(self):
        self.get_number_of_classes()
        self.get_class_probability()
        self.get_class_entropy()
        self.get_class_probability_max()
        self.get_class_probability_mean()
        self.get_class_probability_min()
        self.get_class_probability_std()
        self.get_dataset_ratio()
        self.get_log_dataset_ratio()
        self.get_inverse_dataset_ratio()
        self.get_log_inverse_dataset_ratio()
        return list([self.class_entropy, self.class_probability_max, self.class_probability_mean,
                     self.class_probability_min, self.class_probability_std, self.dataset_ratio,
                     self.inverse_dataset_ratio, self.log_dataset_ratio, self.log_inverse_dataset_ratio,
                     self.number_of_classes])
