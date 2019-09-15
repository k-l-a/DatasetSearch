from MetaFeatures.Output import MetaFeatures

import pandas as pd
from sklearn.datasets import load_svmlight_file
from MetaFeatures.Output import MetaFeatures
from scipy.io import arff
import pandas as pd
import numpy as np
from scipy import spatial
import os
import arff as af


"""
X_train, y_train = load_svmlight_file(data_path)
mat = X_train.todense()
df1 = pd.DataFrame(mat)
df1.columns = range(len(df1.columns))
df2 = pd.DataFrame(y_train)
df2.columns = ['target']
df = pd.concat([df2, df1], axis=1)
df.to_csv("df_data.txt", index=False)
X = df1
y = df2
list = MetaFeatures(X, y).calculate()
print(list)
"""

from MetaFeatures.Output import get_metafeatures


def seperate(df, n):
    df.columns = list(range(n))
    print(n)
    df[n - 1] = df[n - 1].astype(float)
    X = df[list(range(n - 1))]
    y = df[n - 1]
    return X, y

data_path = "D:/code/Similarity/Datasets_/Diabetes.arff"
data = arff.loadarff(data_path)
df = pd.DataFrame(data[0])
n = len(df.columns)
X, y = seperate(df, n)
mf = MetaFeatures(X, y).calculate()

# mf_norm = get_metafeatures(data_path, type="norm")
print(mf)
# print(mf_norm)