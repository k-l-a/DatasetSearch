import autosklearn.classification
from hyperopt import hp, tpe
from hpsklearn import HyperoptEstimator, gradient_boosting, any_classifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", "Mean of empty slice")


path = ""
defaultPath = "./HyperparameterConfig/"
savePath = "./HyperparameterConfig/"
exitCommand = "exit"
loadCommand = "load"
saveCommand = "save"
indexFilename = "./saved_index2"
fileListName = "./saved_filelist2"
indexType = "L2"
fileList = []
searchTerm = ""
weights = []

def getHyperparameter(X, y):
    test_size = int(0.2 * len(y))
    indices = np.random.permutation(X.shape[0])
    X_train = X
    X_test = X
    y_train = y
    y_test = y

    training_idx, test_idx = indices[:80], indices[80:]
    # X_train = X[training_idx,:]

    # y_train = y[indices[:-test_size]]
    # X_test = X[indices[-test_size:]]
    # y_test = y[indices[-test_size:]]

    # estim = HyperoptEstimator(classifier= gradient_boosting('my_gbdt'), max_evals=100, trial_timeout=120)
    automl = autosklearn.classification.AutoSklearnClassifier(
        include_estimators=["gradient_boosting", ], exclude_estimators=None, ensemble_size=1)
    automl.fit(X_train.values, y_train.values)
    automl.cv_results_
    automl.sprint_statistics()
    automl.show_models()
    automl.get_params()

    print(automl.get_params())

    #estim.fit(X_train.values, y_train.values)

    #print(estim.best_model())

def separate(df):
    n = len(df.columns)
    df.columns = list(range(n))
    x = df[list(range(2, n))]
    y = df[1].astype('category')

    return x, y

def catToNum(df):
    df.columns = list(range(len(df.columns)))
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    arr = is_number(df.dtypes)
    for i in range(len(arr)):
        isNum = arr[i]
        if not isNum:
            df[i] = df[i].astype('category').cat.codes

    return df


def getDatasetFromFile(filepath):
    x, y = separate(catToNum(pd.read_csv((filepath))).fillna(0))
    return x, y

def readData():
    folderpath = input("Path to dataset base folder :")
    global path
    if folderpath == loadCommand:
        path = loadCommand
    else:
        path = folderpath

def readDatabaseInput():
    isRead = True
    while isRead:
        try:
            readData()
            X, y = getDatasetFromFile(path)
            getHyperparameter(X, y)
            isRead = False
        except Exception as e:
            print(e)

def main():
    readDatabaseInput()


main()