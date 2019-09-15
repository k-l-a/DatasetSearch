#!/usr/bin/env python
# coding: utf-8

# KeyError: https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/4nIQkZmKdZI

# import
import math
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from sklearn.datasets import load_svmlight_file
import sklearn.metrics
import sklearn.model_selection
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler