import numpy as np
from scipy.stats import kurtosis, skew
from MetaFeatures.AbstractFeature import AbstractFeature


class Kurtosises(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: kurtosis(self.X.iloc[:, x])
        temp = f(temp)
        return temp


"""
class KurtosisMax(AbstractFeature):

    def __init__(self, X, y, *args):
        self.kurtosises = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.kurtosises.max()
"""


class KurtosisMean(AbstractFeature):

    def __init__(self, X, y, *args):
        self.kurtosises = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.kurtosises.mean()


"""
class KurtosisMin(AbstractFeature):

    def __init__(self, X, y, *args):
        self.kurtosises = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.kurtosises.min()
"""


class KurtosisStd(AbstractFeature):

    def __init__(self, X, y, *args):
        self.kurtosises = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.kurtosises.std()


class Skewnesses(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: skew(self.X.iloc[:, x])
        temp = f(temp)
        return temp


"""
class SkewnessMax(AbstractFeature):

    def __init__(self, X, y, *args):
        self.skewnesses = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.skewnesses.max()
"""


class SkewnessMean(AbstractFeature):

    def __init__(self, X, y, *args):
        self.skewnesses = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.skewnesses.mean()


"""
class SkewnessMin(AbstractFeature):

    def __init__(self, X, y, *args):
        self.skewnesses = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.skewnesses.min()
"""


class SkewnessStd(AbstractFeature):

    def __init__(self, X, y, *args):
        self.skewnesses = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.skewnesses.std()


class Symbols(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: np.unique(self.X.iloc[:, x])
        g = lambda x: np.sum(np.isfinite(x))
        return np.array([g(f(x)) for x in temp])


"""
class SymbolsMax(AbstractFeature):

    def __init__(self, X, y, *args):
        self.symbols = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.symbols.max()
"""


class SymbolsMean(AbstractFeature):

    def __init__(self, X, y, *args):
        self.symbols = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.symbols.mean()


"""
class SymbolsMin(AbstractFeature):

    def __init__(self, X, y, *args):
        self.symbols = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.symbols.min()
"""


class SymbolsStd(AbstractFeature):

    def __init__(self, X, y, *args):
        self.symbols = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.symbols.std()


class SymbolsSum(AbstractFeature):

    def __init__(self, X, y, *args):
        self.symbols = args[0]
        super().__init__(X, y)

    def calculate(self):
        return self.symbols.sum()


class DistributionFeatures(AbstractFeature):

    def __init__(self,X, y, *args):
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
        super().__init__(X, y)

    def calculate(self):
        self.kurtosises = Kurtosises(self.X, self.y).value
        self.skewnesses = Skewnesses(self.X, self.y).value
        self.symbols = Symbols(self.X, self.y).value
        # self.kurtosis_max = KurtosisMax(self.X, self.y, self.kurtosises).value
        self.kurtosis_mean = KurtosisMean(self.X, self.y, self.kurtosises).value
        # self.kurtosis_min = KurtosisMin(self.X, self.y, self.kurtosises).value
        self.kurtosis_std = KurtosisStd(self.X, self.y, self.kurtosises).value
        # self.skewness_max = SkewnessMax(self.X, self.y, self.skewnesses).value
        self.skewness_mean = SkewnessMean(self.X, self.y, self.skewnesses).value
        # self.skewness_min = SkewnessMin(self.X, self.y, self.skewnesses).value
        self.skewness_std = SkewnessStd(self.X, self.y, self.skewnesses).value
        # self.symbols_max = SymbolsMax(self.X, self.y, self.symbols).value
        self.symbols_mean = SymbolsMean(self.X, self.y, self.symbols).value
        # self.symbols_min = SymbolsMin(self.X, self.y, self.symbols).value
        self.symbols_std = SymbolsStd(self.X, self.y, self.symbols).value
        self.symbols_sum = SymbolsSum(self.X, self.y, self.symbols).value
        return [self.kurtosis_mean,
                     self.kurtosis_std, self.skewness_mean,
                     self.skewness_std,
                     self.symbols_mean, self.symbols_std,
                     self.symbols_sum]
