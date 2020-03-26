import autosklearn.classification
import autosklearn.pipeline.components.classification.gradient_boosting
import sklearn
from sklearn.model_selection import train_test_split
from MetaFeatures.Output import MetaFeatures
from MetaFeatures import Output
import json
import sys
import os
import csv
import numpy as np
import pandas as pd
import warnings
import multiprocessing
from multiprocessing import Pool
import shutil


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import MULTICLASS_CLASSIFICATION
warnings.filterwarnings("ignore", "Mean of empty slice")

defaultFile = "./WorkingDataset/"
path = ""
defaultPath = "./HyperparameterConfig/"
savePath = "./HPConfig2/"
dataPath = "./MetaConfigData2/"
accPath = "./Accuracies2/"
exitCommand = "exit"
loadCommand = "load"
saveCommand = "save"
indexFilename = "./saved_index2"
fileListName = "./saved_filelist2"
configFileName = ""
accFileName = ""
saveFileName = ""
resultFileName = "1result.csv"
indexType = "L2"
fileList = []
searchTerm = ""
weights = []

hyperParamDict = {
    # add all relevant hyperparams here
    # 'balancing:strategy': "none",
    # 'classifier:gradient_boosting:early_stop': 'off',
    'classifier:gradient_boosting:l2_regularization': 0.1,
    'classifier:gradient_boosting:learning_rate' : 0.1,
    #'classifier:gradient_boosting:loss' : "deviance",
    'classifier:gradient_boosting:max_bins': 10,
    'classifier:gradient_boosting:max_depth' : 3,
    'classifier:gradient_boosting:max_iter' : 1,
    'classifier:gradient_boosting:max_leaf_nodes' : 0,
    'classifier:gradient_boosting:min_samples_leaf' : 1,
    # 'classifier:gradient_boosting:scoring': 'loss',
    'classifier:gradient_boosting:tol': 0.1,
    # 'classifier:gradient_boosting:n_iter_no_change': 10,
    # 'classifier:gradient_boosting:validation_fraction': 0.1
    }

is_default = False
read_command_line = False
find_hyperparams = True
make_config_dataset = True
read_config = False
use_parallel = True
base_model = "libsvm_svc"


def spawn_classifier(dataset_names):
    """Spawn a subprocess.

    auto-sklearn does not take care of spawning worker processes. This
    function, which is called several times in the main block is a new
    process which runs one instance of auto-sklearn.
    """
    print(dataset_names)
    # Use the initial configurations from meta-learning only in one out of
    # the four processes spawned. This prevents auto-sklearn from evaluating
    # the same configurations in four processes.
    for dataset_name in dataset_names:
        try:
            configFileName = dataset_name[:-4] + "_params.json"
            accFileName = dataset_name[:-4] + "_acc.txt"
            X, y = getDatasetFromFile(path + dataset_name)
            getHyperparameter(X, y, configFileName, accFileName, 42, dataset_name)
        except Exception as e:
            print("ERROR!")
            print(e)


def getHyperparameterParallel(path):
    filenames = []
    for filename in os.listdir(path):
        filenames.append(filename)


    core = int(os.cpu_count()/2)
    file_per_core = int(len(filenames)/core)

    processes = []
    for i in range(core):  # set this at roughly half of your cores
        files = []
        for j in range(file_per_core):
            n = i * file_per_core + j
            if n < len(filenames):
                files.append(filenames[n])

        p = multiprocessing.Process(
            target=spawn_classifier,
            args=[files],
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def getHyperparameter(X, y, configFileName, accFileName, seed, dataset):
    test_size = int(0.2 * len(y))
    indices = np.random.permutation(X.shape[0])
    X_train = X
    y_train = y
    X_test = X
    y_test = y


    training_idx, test_idx = indices[:80], indices[80:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("test")
    automl = autosklearn.classification.AutoSklearnClassifier(
        seed=seed, include_estimators=[base_model, ], ensemble_size=1,
        time_left_for_this_task=60, ml_memory_limit=1024, per_run_time_limit=30)
    automl.fit(X_train, y_train, dataset_name=dataset)
    # automl.cv_results_
    # print("sprint:")
    # automl.sprint_statistics()
    # print("models:")
    # automl.show_models()
    y_hat = automl.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    saveAcc(acc, accPath + accFileName)
    # print("Accuracy: ", acc)
    params = automl.get_params()
    print("params:")
    print(params)
    print("Models:")
    #print(automl.show_models())
    gbdt = automl.show_models()
    saveJson(gbdt, savePath + configFileName)


def saveJson(params, filename):
    with open(filename, 'w') as outfile:
        json.dump(params, outfile)

def saveAcc(acc, filename):
    with open(filename, 'w') as outfile:
        outfile.write(str(acc))

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

def getMetafeatures(x, y):
    featureList = []

    metaFeatures = MetaFeatures(x, y).calculate()

    d = len(metaFeatures)
    print(d)
    return metaFeatures


def makeConfigDataset(metafeatures, hyperparams):
    if len(metafeatures) != len(hyperparams):
        print("Datapoints of metafeatures and hyperparams are not equal in number!")
        print("Meta #: ", len(metafeatures))
        print("HP #: ", len(hyperparams))

    for i in range(len(metafeatures)):
        curr_meta = metafeatures[i]
        curr_hp = hyperparams[i]


def readData():
    if is_default:
        folderpath = defaultFile
    elif read_command_line:
        folderpath = sys.argv[1]
    else:
        folderpath = input("Path to dataset base folder :")
    global path, configFileName
    if folderpath == loadCommand:
        path = loadCommand
    else:
        path = folderpath

def processDatabase(path, mode="regression"):
    global configFileName, accFileName, saveFileName
    featureList = []
    resultDF = pd.DataFrame()
    first = True
    print("test")
    for filename in os.listdir(path):
        try:
            print(filename)
            configFileName = filename[:-4] + "_params.json"
            accFileName = filename[:-4] + "_acc.txt"
            saveFileName = filename[:-4] + "_meta_config"
            X, y = getDatasetFromFile(path + filename)
            if find_hyperparams:
                getHyperparameter(X, y, configFileName, accFileName, 42, filename)
            elif make_config_dataset:
                print('test3')
                print(X)
                print(y)
                metaFeatures = getMetafeatures(X, y)
                print('test2')
                featureList.append(metaFeatures)
                hp = [1]
                print(hp)
                if len(hp) == 0:
                    continue
                df = pd.DataFrame(metaFeatures).T
                if mode == "regression":
                    saveFileName += "_regression.csv"
                    df[df.shape[1]] = [hp]
                    print(type([hp]))
                elif mode == "classification":
                    saveFileName += "_classification.csv"
                    base = df.shape[1]
                    for i in range(len(hp)):
                        df[base + i] = hp[i]
                else:
                    print("unknown type of evaluation (not regression/classification)")

                df.to_csv(dataPath + saveFileName)
                if first:
                    resultDF = df
                    first = False
                else:
                    resultDF = resultDF.append(df, ignore_index=True)

            fileList.append(filename)
        except Exception as e:
            print(e)
            print("Could not get hyperparameters / metafeatures of " + filename)
            continue

def convertTextToConfig(text):
    temp = text
    vec = temp.split("{")[1].split("}")
    hyperp = vec[0]
    datasetp = vec[1]
    hyperp = hyperp.split(",")
    values = []
    for i in range(0, (len(hyperp))):
        tmp = hyperp[i].split(":")
        name = ""
        for j in range(0, len(tmp) - 1):
            name += tmp[j]
            name += ":"
        name = name[:-1]
        val = tmp[-1]
        # Value found.
        # remove '' and whitespace
        name = name.strip()
        name = name[1:-1]
        val = val.strip()
        print(name)
        print(val)
        if name in hyperParamDict:
            default = hyperParamDict[name]
            if val == "'None'":
                val = 1000
            elif isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
            else:
                val = val[1:-1]
            values.append(val)

    return values


def readHyperParamFromFile(filename):
    values = []
    try:
        conf = open(savePath + configFileName)
        values = [1]
    except Exception as e:
        print("Cannot find hyperparams for " +  configFileName + ".")
        print(e)
    return values


def readDatabaseInput():
    isRead = True
    while isRead:
        try:
            readData()
            if read_config:
                conf = open(path)
                values = convertTextToConfig(conf.read())
                print(values)
            elif find_hyperparams and use_parallel:
                isRead = False
                getHyperparameterParallel(path)
            else:
                isRead = False
                processDatabase(path)
            isRead = False
        except Exception as e:
            print(e)

def filter_file(sourceDir, destDir, compDir):
    for filename in os.listdir(compDir):
        conf = open(compDir + filename)
        vec = conf.read().split("{")
        if len(vec) > 1:
            original = filename[:-12]
            original = original + ".csv"
            if os.path.exists(sourceDir + original):
                shutil.copyfile(sourceDir + original, destDir + original)




def main():
    readDatabaseInput()

main()
