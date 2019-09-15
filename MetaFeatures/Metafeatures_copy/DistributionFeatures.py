import numpy as np
from scipy.stats import kurtosis, skew


class DistributionFeatures:

    def __init__(self, basic_features):
        self.basic_features = basic_features
        self.X = basic_features.X
        self.y = basic_features.y
        self.kurtosis_max = None
        self.kurtosis_mean = None
        self.kurtosis_min = None
        self.kurtosis_std = None
        self.skewness_max = None
        self.skewness_mean = None
        self.skewness_min = None
        self.skewness_std = None
        self.symbols_max = None
        self.symbols_mean = None
        self.symbols_min = None
        self.symbols_std = None
        self.symbols_sum = None
        self.kurtosises = None
        self.skewnesses = None
        self.symbols = None

    # kurtosis-max
    # first calculate kurtosis of all features
    def get_kurtosises(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: kurtosis(self.X[x])
        temp = f(temp)
        self.kurtosises = temp
        return self.kurtosises

    # kurtosis-max
    def get_kurtosis_max(self):
        if self.kurtosises is None:
            self.get_kurtosises()
        self.kurtosis_max = self.kurtosises.max()
        return self.kurtosis_max

    # kurtosis-mean
    def get_kurtosis_mean(self):
        if self.kurtosises is None:
            self.get_kurtosises()
        self.kurtosis_mean = self.kurtosises.mean()
        return self.kurtosis_mean

    # kurtosis-min
    def get_kurtosis_min(self):
        if self.kurtosises is None:
            self.get_kurtosises()
        self.kurtosis_min = self.kurtosises.min()
        return self.kurtosis_min

    # kurtosis-std
    def get_kurtosis_std(self):
        if self.kurtosises is None:
            self.get_kurtosises()
        self.kurtosis_std = self.kurtosises.std()
        return self.kurtosis_std

    # skewness
    def get_skewnesses(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: skew(self.X[x])
        temp = f(temp)
        self.skewnesses = temp
        return self.skewnesses

    # skewness-max
    def get_skewness_max(self):
        if self.skewnesses is None:
            self.get_skewnesses()
        self.skewness_max = self.skewnesses.max()
        return self.skewness_max

    # skewness-mean
    def get_skewness_mean(self):
        if self.skewnesses is None:
            self.get_skewnesses()
        self.skewness_mean = self.skewnesses.mean()
        return self.skewness_mean

    # skewness-min
    def get_skewness_min(self):
        if self.skewnesses is None:
            self.get_skewnesses()
        self.skewness_min = self.skewnesses.min()
        return self.skewness_min

    # skewness-std
    def get_skewness_std(self):
        if self.skewnesses is None:
            self.get_skewnesses()
        self.skewness_std = self.skewnesses.std()
        return self.skewness_std

    # symbols-max
    def get_symbols(self):
        temp = np.arange(self.basic_features.number_of_features)
        f = lambda x: np.unique(self.X[x])
        g = lambda x: np.sum(np.isfinite(x))
        self.symbols = g(f(temp))
        return self.symbols

    def get_symbols_max(self):
        if self.symbols is None:
            self.get_symbols()
        self.symbols_max = self.symbols.max()
        return self.symbols_max

    # symbols-sum
    def get_symbols_sum(self):
        if self.symbols is None:
            self.get_symbols()
        self.symbols_sum = self.symbols.sum()
        return self.symbols_sum

    # symbols-mean
    def get_symbols_mean(self):
        if self.symbols is None:
            self.get_symbols()
        self.symbols_mean = self.symbols.mean()
        return self.symbols_mean

    # symbols-min
    def get_symbols_min(self):
        if self.symbols is None:
            self.get_symbols()
        self.symbols_min = self.symbols.min()
        return self.symbols_min

    # symbols-std
    def get_symbols_std(self):
        if self.symbols is None:
            self.get_symbols()
        self.symbols_std = self.symbols.std()
        return self.symbols_std

    def calculate(self):
        self.get_kurtosises()
        self.get_kurtosis_max()
        self.get_kurtosis_mean()
        self.get_kurtosis_min()
        self.get_kurtosis_std()
        self.get_skewnesses()
        self.get_skewness_max()
        self.get_skewness_mean()
        self.get_skewness_min()
        self.get_skewness_std()
        self.get_symbols()
        self.get_symbols_max()
        self.get_symbols_mean()
        self.get_symbols_min()
        self.get_symbols_std()
        self.get_symbols_sum()
        return list([self.kurtosis_max, self.kurtosis_mean, self.kurtosis_min,
                     self.kurtosis_std, self.skewness_max, self.skewness_mean,
                     self.skewness_min, self.skewness_std, self.symbols_max,
                     self.symbols_mean, self.symbols_min, self.symbols_std,
                     self.symbols_sum])
