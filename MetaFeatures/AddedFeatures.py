import numpy as np
from scipy.stats import entropy, shapiro
from MetaFeatures.AbstractFeature import AbstractFeature
# from scipy.stats.mstats_basic import f_value_wilks_lambda


class AttrEnt(AbstractFeature):
    
    def __init__(self, X, y, *args):
        super().__init__(X, y)
    
    def calculate(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: self.X.iloc[:, x]
        temp = f(temp)
        g = lambda feature : np.unique(self.y, return_counts=True)[1]
        temp = [g(i) for i in temp]
        return np.array([entropy(i) for i in temp])


class AttrEntMean(AbstractFeature):
    
    def __init__(self, X, y, *args):
        self.attr_ent = args[0]
        super().__init__(X, y)
    
    def calculate(self):
        return self.attr_ent.mean()


class AttrEntStd(AbstractFeature):
    
    def __init__(self, X, y, *args):
        self.attr_ent = args[0]
        super().__init__(X, y)
    
    def calculate(self):
        return self.attr_ent.std()


class InstToAttr(AbstractFeature):
    
    def __init__(self, X, y, *args):
        super().__init__(X, y)
    
    def calculate(self):
        return len(self.y) / len(self.X.columns)


class AttrToInst(AbstractFeature):
    
    def __init__(self, X, y, *args):
        super().__init__(X, y)
    
    def calculate(self):
        return len(self.X.columns) / len(self.y)


"""
class WilksLambda(AbstractFeature):
    
    def __init__(self,X, y, *args):
        super().__init__(X, y)
    
    def calculate(self):
        return f_value_wilks_lambda()
"""


class AttrVar(AbstractFeature):
    
    def __init__(self, X, y, *args):
        super().__init__(X, y)
    
    def calculate(self):
        temp = np.arange(len(self.X.columns))
        f = lambda x: self.X.iloc[:, x]
        temp = [f(i) for i in temp]
        return np.array([i.var() for i in temp])


class AttrVarMean(AbstractFeature):
    
    def __init__(self, X, y, *args):
        self.attr_var = args[0]
        super().__init__(X, y)
    
    def calculate(self):
        return self.attr_var.mean()


class AttrVarStd(AbstractFeature):
    
    def __init__(self, X, y, *args):
        self.attr_var = args[0]
        super().__init__(X, y)
    
    def calculate(self):
        return self.attr_var.std()


class PerOfNormAttr(AbstractFeature):
    
    def __init__(self, X, y, *args):
        super().__init__(X, y)
    
    def calculate(self):
        temp = np.arange(len(self.X.columns))
        count = 0
        for i in temp:
            w, p = shapiro(self.X.iloc[:, i:i + 1])
            if w > 0.9 and p < 0.1:
                count = count + 1
        return count / len(self.X.columns)


class AdFeatures(AbstractFeature):
    
    def __init__(self, X, y, *args):
        self.attr_ent = None
        self.attr_ent_mean = None
        self.attr_ent_std = None
        self.inst_to_attr = None
        self.attr_to_inst = None
        self.attr_var = None
        self.attr_var_mean = None
        self.attr_var_std = None
        self.per_of_norm_attr = None
        super().__init__(X, y)
    
    def calculate(self):
        self.attr_ent = AttrEnt(self.X, self.y).value
        self.attr_ent_mean = self.attr_ent.mean()
        self.attr_ent_std = self.attr_ent.std()
        self.inst_to_attr = InstToAttr(self.X, self.y).value
        self.attr_to_inst = AttrToInst(self.X, self.y).value
        self.attr_var = AttrVar(self.X, self.y).value
        self.attr_var_mean = self.attr_var.mean()
        self.attr_var_std = self.attr_var.std()
        self.per_of_norm_attr = PerOfNormAttr(self.X, self.y).value
        return [self.attr_ent_mean,
                self.attr_ent_std,
                self.inst_to_attr,
                self.attr_to_inst,
                self.attr_var_mean,
                self.attr_var_std,
                self.per_of_norm_attr]