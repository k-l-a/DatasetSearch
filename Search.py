import numpy as np
import autosklearn.classification
import autosklearn.regression
import autosklearn.pipeline.components.classification.gradient_boosting
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn
import warnings
import faiss
import CGBDTClassifier
from sklearn.model_selection import train_test_split
from MetaFeatures.Output import MetaFeatures
from MetaFeatures import Output
import WriteHP as whp
from dataset2vec import main as d2v
from dataset2vec import config
from enum import Enum
from functools import partial
import multiprocessing
import ConvertCsvtoSvmLight
import math
import os
import pandas as pd
import pickle
import csv
from pathlib import Path


warnings.filterwarnings("ignore", "Mean of empty slice")

path = ""
defaultPath = "./ManualTestDatasets/"
savePath = "./SavedDatasets/"
exitCommand = "exit"
loadCommand = "load"
saveCommand = "save"
indexFilename = "./saved_index"
fileListName = "./saved_filelist"
modelName = "./saved_model_svm_1h"
indexType = "L2"
fileList = []
searchTerm = ""
weights = []
codes = []
hyperParamDict = {
    # add all relevant hyperparams here
    'balancing:strategy': "none",
    'classifier:gradient_boosting:early_stop': 'off',
    'classifier:gradient_boosting:l2_regularization': 0.1,
    'classifier:gradient_boosting:learning_rate' : 0.1,
    'classifier:gradient_boosting:loss' : "deviance",
    'classifier:gradient_boosting:max_bins': 10,
    'classifier:gradient_boosting:max_depth' : 3,
    'classifier:gradient_boosting:max_iter' : 1,
    'classifier:gradient_boosting:max_leaf_nodes' : 0,
    'classifier:gradient_boosting:min_samples_leaf' : 1,
    'classifier:gradient_boosting:scoring': 'loss',
    'classifier:gradient_boosting:tol': 0.1,
    # 'classifier:gradient_boosting:n_iter_no_change': 10,
    # 'classifier:gradient_boosting:validation_fraction': 0.1
    }
hyperParamDictSVM = {
    'classifier:libsvm_svc:C':1.0,
    'classifier:libsvm_svc:degree':3,
    'classifier:libsvm_svc:gamma':0.1,
    'classifier:libsvm_svc:kernel':"rbf",
    'classifier:libsvm_svc:max_iter': -1,
    'classifier:libsvm_svc:shrinking': True,
    'classifier:libsvm_svc:tol': 1e-3,
    'classifier:libsvm_svc:coef0': 0.0
    }



usedHP = [0,1,4,5,6]
usedHPSVM = [0,1,2,3,4]
testingAccPath = "./TestingAcc/"
trainingAccPath = "./TrainingAcc/"
tempSavePath = "./Temp/"
mode = "regression"
baseModel = "svm"

useDefault = False
useCSV = True
used2v = False
fromCsv = False
isWeighted = False
D2C = True
useParallel = True
useTemp = True
duration = 60


d = 46
k = 5


class MeasureType(Enum):
    L2 = 1
    COS = 2
    MANH = 3
    MINK = 4


def readData():
    folderpath = input("Path to dataset base folder :")
    global path
    if folderpath == loadCommand:
        path = loadCommand
    elif useDefault:
        path = defaultPath
    else:
        path = folderpath


def separate(df):
    n = len(df.columns)
    df.columns = list(range(n))
    if D2C:
        if mode == "classification":
            x = df[list(range(1, d+1))]
            y = df[range(d+1, n)].astype('category')
        elif mode == "regression":
            x = df[list(range(1, n-1))]
            y = df[n-1]
    else:
        x = df[list(range(2, n))]
        y = df[1].astype('category')

    print(y)

    return x, y


def catToNum(df):
    if mode == "classification":
        df.columns = list(range(len(df.columns)))
        is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
        arr = is_number(df.dtypes)
        for i in range(len(arr)):
            isNum = arr[i]
            if not isNum:
                dic = dict(enumerate(df[i].astype('category').cat.categories))
                codes.append(dic)
                df[i] = df[i].astype('category').cat.codes

    return df


def getMetafeatures():
    featureList = []
    global d
    for filename in os.listdir(path):
        try:
            print(filename)
            actualPath = path
            if useCSV:
                x, y = separate(catToNum(pd.read_csv((path + filename))).fillna(0))
            elif fromCsv:
                ConvertCsvtoSvmLight.convert((path + filename), (savePath + filename), 1, 1)
                actualPath = savePath

            if useCSV:
                metaFeatures = MetaFeatures(x, y).calculate()
            else:
                metaFeatures = Output.get_metafeatures(actualPath + filename)

            if used2v:
                d2vFeatures = d2v.main(config=config, dataset=path + filename, seed=0)
                for i in d2vFeatures:
                    metaFeatures.append(i)

            d = len(metaFeatures)
            print(d)
            featureList.append(metaFeatures)
            fileList.append(filename)
        except Exception as e:
            print("Could not get metafeatures of " + filename)
            print(e)
            continue

    arr = np.array(featureList).astype('float32')
    print(arr.shape)
    return arr


# Function to process dataset. Accepts a dataset (x, y) and returns a 'dataset'(x', y') that has been processed
def processDataset(x, y, model):
    if D2C:
        # get metafeatures
        metafeatures = MetaFeatures(x, y).calculate()

        # Train gdbt model and get hyperparams
        # TBD : Use dataset2config
        model = autosklearn.classification.AutoSklearnClassifier(include_estimators=["gradient_boosting", ],
                                                                 exclude_estimators=None, ensemble_size=1)
        model.fit(x, y)
        hyperparams = model.get_params()
        return metafeatures, hyperparams
    else:
        # Default: no processing done
        return x, y


def similarityModel(x, y, model, load=True):
    if D2C:
        model.fit(x, y)
        return model
    else:
        # Default: create Faiss index
        if load:
            model = loadIndex(indexFilename)
        else:
            model = faiss.IndexFlatL2(d)
        return model


def normalizeDimension(vector):
    if len(vector) < d:
        for i in range(d - len(vector)):
            vector.append(0)  # Pad the features to reach desired dimension
    else:
        vector = vector[:d]
    return vector


def normalizeMeta(vector, weights):
    if len(vector) != len(weights):
        print("Different dimension of vector and weights")
    else:
        for i in range(len(vector)):
            vector[i] = vector[i] * math.sqrt(weights[i])


def loadWeights(inputWeights):
    if len(inputWeights) != d:
        print("Error, expected " + d + " values, received " + len(inputWeights) + " values.")
    else:
        weights = inputWeights
        isWeighted = True


def loadIndex(filename):
    return faiss.read_index(filename)


def readSearchTerm():
    term = input("Dataset to use as search term :")
    global searchTerm
    if term == "exit":
        searchTerm = "exit"
    elif useDefault:
        searchTerm = defaultPath + term
    else:
        searchTerm = term


def search(termPath, metaFeatureList, load=True):
    global d, k
    if load:
        index = loadIndex(indexFilename)
    else:
        index = faiss.IndexFlatL2(d)
        index.add(metaFeatureList)
    print(index.is_trained)
    if termPath == saveCommand:
        faiss.write_index(index, indexFilename)
        with open(fileListName, "wb") as fp:
            pickle.dump(fileList, fp)
        print("Index and file list saved")
    else:
        if useCSV:
            x, y = separate(catToNum(pd.read_csv((termPath))))
            termFeatures = MetaFeatures(x, y).calculate()
        else:
            termFeatures = Output.get_metafeatures(termPath)
        termFeatures = np.array([termFeatures]).astype('float32')
        D, I = index.search(termFeatures, k)
        print(I[:5])
        for i in I[:5]:
            for j in i:
                print(fileList[j])


def readDatabaseInput():
    isRead = True
    while isRead:
        try:
            readData()
            if path == loadCommand:
                meta = []
                with open(fileListName, "rb") as fp:
                    global fileList
                    fileList = pickle.load(fp)
            elif D2C:
                meta = handleD2C()
                return meta
            else:
                meta = getMetafeatures()
            isRead = False
        except Exception as e:
            print("failed to read")
            print(e)
    return meta


def readSearchInput():
    try:
        readSearchTerm()
        if searchTerm == "exit":
            return False
        return True
    except Exception as e:
        print(e)




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



def createClassificationModel():
    model =  autosklearn.classification.AutoSklearnClassifier(
        seed=42, include_estimators=["gradient_boosting", ],
        time_left_for_this_task=600, per_run_time_limit=60)

    return model

def createRegressionModel():
    model =  autosklearn.regression.AutoSklearnRegressor(
        seed=42, include_estimators=["gradient_boosting", ],
        time_left_for_this_task=3600, per_run_time_limit=60)

    return model

def getVectorFromStr(vecstr):
    vec = []
    str = vecstr.split(",")
    str[0] = str[0][1:]
    str[-1] = str[-1][:-1]
    vec.append(float(str[0]))
    vec.append(float(str[1]))
    vec.append(int(str[2]))
    vec.append(int(str[3]))
    vec.append(int(str[4]))
    vec.append(int(str[5]))
    vec.append(float(str[6]))
    vec.append(float(str[7]))

    return vec

def getVectorFromStrSvm(vecstr):
    vec = []
    str = vecstr.split(",")
    str[0] = str[0][1:]
    str[-1] = str[-1][:-1]
    vec.append(float(str[0]))
    vec.append(float(str[1]))
    vec.append(float(str[3]))
    vec.append(float(str[4]))
    vec.append(int(str[5]))

    return vec

def SplitVectorDataframe(vectordf, n):
    result = []
    for i in range(n):
        tmp = vectordf.apply(lambda v : v[i])
        result.append(tmp)

    return result


def handleD2C():
    ys_train = []
    ys_test = []
    models = []
    train_acc = []
    test_acc = []
    global d
    for filename in os.listdir(path):
        try:
            print(filename)
            file = open(path+filename)
            if useCSV:
                x, y = separate(catToNum(pd.read_csv((path + filename), skipinitialspace=True,
                                                    sep=',', quotechar='"')))

            print(y)
            if baseModel == "gbdt":
                y = y.apply(getVectorFromStr)
            elif baseModel == "svm":
                y = y.apply(getVectorFromStrSvm)
            print("Success")
            # mlb = MultiLabelBinarizer()
            # y = mlb.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            feature_types = (['categorical'] * 2) + (['numerical'] * 2) + ['categorical'] + (['numerical'] * 5) + [
                'categorical'] + ['numerical']
            model = createRegressionModel()
            length = 0
            split_y_train = []
            split_y_test = []
            if mode == "classification":
                model = createClassificationModel()
                length = len(y_train.columns)
            elif mode == "regression":
                length = len(y_train.iloc[0])
                print(y_train.iloc[0])
                print(length)
                split_y_train = SplitVectorDataframe(y_train, length)
                split_y_test = SplitVectorDataframe(y_test, length)

            if useParallel:
                models, train_acc, test_acc = regression_parallel(split_y_train, split_y_test, ys_train,
                                                               ys_test, X_train, X_test, test_acc, train_acc, filename)
            else:
                for i in range(length):
                    if mode == "classification":
                        model = createClassificationModel()
                        curr_y_train = y_train.iloc[:, i]
                        curr_y_test = y_test.iloc[:, i]
                    elif mode == "regression":
                        model = createRegressionModel()
                        curr_y_train = split_y_train[i]
                        curr_y_test = split_y_test[i]
                        print(curr_y_train)
                    ys_train.append(curr_y_train)
                    ys_test.append(curr_y_test)
                    model.fit(X_train, curr_y_train)
                    models.append(model)
                    y_hat = model.predict(X_test)
                    np.savetxt(testingAccPath + filename[:-4] + "_testing_res_" + str(i) + ".txt", y_hat)
                    if mode == "classification":
                        acc = sklearn.metrics.accuracy_score(curr_y_test, y_hat)
                    else:
                        acc = sklearn.metrics.r2_score(curr_y_test, y_hat)
                    print(acc)
                    test_acc.append(acc)
                    y_hat = model.predict(X_train)
                    if mode == "classification":
                        acc = sklearn.metrics.accuracy_score(curr_y_train, y_hat)
                    else:
                        acc = sklearn.metrics.r2_score(curr_y_train, y_hat)
                    print(acc)
                    train_acc.append(acc)

            fileList.append(filename)
        except Exception as e:
            print("Could not create model from  " + filename)
            print(e)
            continue

    # Testing error
    with open(testingAccPath + filename[:-4] + "_testing_acc.txt", 'w') as outfile:
        for acc in test_acc:
            outfile.write(str(acc))
            outfile.write("\n")

    # Training error
    with open(trainingAccPath + filename[:-4] + "_training_acc.txt", 'w') as outfile:
        for acc in train_acc:
            outfile.write(str(acc))
            outfile.write("\n")

    print("test acc:")
    for acc in test_acc:
        print(acc)

    print("train_acc:")
    for acc in train_acc:
        print(acc)

    with open(modelName, "wb") as fp:
        pickle.dump(models, fp)

    with open("code.txt", "wb") as fp:
        pickle.dump(codes, fp)

    return models

def regression_parallel(split_y_train, split_y_test, ys_train, ys_test, X_train, X_test, test_acc, train_acc, filename):
    print(filename)
    print(X_test)
    model = createRegressionModel()
    models = []

    for i in range(5):
        curr_y_train = split_y_train[i]
        curr_y_test = split_y_test[i]
        ys_train.append(curr_y_train)
        ys_test.append(curr_y_test)
        models.append(createRegressionModel())

    processes = []
    func = partial(helper_regression, models, split_y_train, X_train)

    for i in range(5):  # set this at roughly half of your cores

        p = multiprocessing.Process(
            target=func,
            args=[i],
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(models)
    print(len(models))

    hps = []
    if baseModel == "gbdt":
        hps = usedHP
    elif baseModel == "svm":
        hps = usedHPSVM

    for i in range(5):
        if i in hps:
            model = models[i]
            print("TESTMODEL")
            print(type(model))
            y_hat = model.predict(X_test)
            print("TESTTTT")
            np.savetxt(testingAccPath + filename[:-4] + "_testing_res_" + str(i) + ".txt", y_hat)
            if mode == "classification":
                acc = sklearn.metrics.accuracy_score(curr_y_test, y_hat)
            else:
                acc = sklearn.metrics.r2_score(curr_y_test, y_hat)
            print(acc)
            test_acc.append(acc)
            y_hat = model.predict(X_train)
            if mode == "classification":
                acc = sklearn.metrics.accuracy_score(curr_y_train, y_hat)
            else:
                acc = sklearn.metrics.r2_score(curr_y_train, y_hat)
            print(acc)
            train_acc.append(acc)

    return models, train_acc, test_acc


def helper_regression(models, split_y_train, X_train, i):
    model = createRegressionModel()
    curr_y_train = split_y_train[i]
    print(curr_y_train)
    model.fit(X_train, curr_y_train)
    models[i] = model


def fit_custom_and_default(path, filename):
    x, y = separate(catToNum(pd.read_csv((path + filename), skipinitialspace=True)))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    automl = autosklearn.classification.AutoSklearnClassifier(
        seed=42, include_estimators=['gradient_boosting'], ensemble_size=1,
        initial_configurations_via_metalearning=0,
        time_left_for_this_task=60, per_run_time_limit=30)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    print("testing:")
    print(acc)
    with open("./TestingAcc/" + filename[:-4] + "_old_test_acc_1.txt", 'w') as outfile:
        outfile.write(str(acc))
    y_hat = automl.predict(X_train)
    acc = sklearn.metrics.accuracy_score(y_train, y_hat)
    with open("./TrainingAcc/" + filename[:-4] + "_old_train_acc_1.txt", 'w') as outfile:
        outfile.write(str(acc))

    print("training:")
    print(acc)

    print("DIVIDER")

    print("Done!")
    print(automl.show_models())
    hp = convertTextToConfig(automl.show_models())
    print(hp)
    with open("./ManualTesting/FinalHP/" + filename[:-4] + "_old_final_hp_1.txt", 'w') as outfile:
        outfile.write(str(hp))


def spawn_classifier(X_train, X_test, y_train, y_test, filename, i):
    automl = autosklearn.classification.AutoSklearnClassifier(
        seed=42, include_estimators=['libsvm_svc', ], ensemble_size=1,
        initial_configurations_via_metalearning=0,
        time_left_for_this_task=(i + 1) * 60, per_run_time_limit=30)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    print(acc)
    dir = "./TestingAcc/"
    if useTemp:
        dir = tempSavePath + dir[2:]
        Path(dir).mkdir(parents=True, exist_ok=True)

    with open(dir + "parallel_" + filename[:-4] + "cust_test_acc_1h_" + str(i) + ".txt", 'w') as outfile:
        outfile.write(str(acc))
    y_hat = automl.predict(X_train)
    acc = sklearn.metrics.accuracy_score(y_train, y_hat)
    print(acc)

    dir = "./TrainingAcc/"
    if useTemp:
        dir = tempSavePath + dir[2:]
        Path(dir).mkdir(parents=True, exist_ok=True)

    with open(dir + "parallel_" + filename[:-4] + "cust_train_acc_1h_" + str(i) + ".txt", 'w') as outfile:
        outfile.write(str(acc))

    hp = convertTextToConfig(automl.show_models())
    print(hp)

    dir = "./ManualTesting/FinalHP/"
    if useTemp:
        dir = tempSavePath + dir[2:]
        Path(dir).mkdir(parents=True, exist_ok=True)

    with open(dir + "parallel_" + filename[:-4] + "cust_hp_1h_" + str(i) + ".txt", 'w') as outfile:
        outfile.write(str(hp))


def fit_and_save_over_time(path, filename):
    x, y = separate(catToNum(pd.read_csv((path + filename), skipinitialspace=True)))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if useParallel:

        core = int(os.cpu_count() / 2)
        processes = []
        core = duration

        for i in range(core):  # set this at roughly half of your cores
            if i < duration:
                p = multiprocessing.Process(
                    target=spawn_classifier,
                    args=(X_train, X_test, y_train, y_test, filename, i),
                )
                p.start()
                processes.append(p)
        for p in processes:
            p.join()
    else:
        for i in range(duration):
            automl = autosklearn.classification.AutoSklearnClassifier(
                seed=42, include_estimators=['libsvm_svc', ], ensemble_size=1,
                initial_configurations_via_metalearning=0,
                time_left_for_this_task=(i+1) * 60, per_run_time_limit=30)
            automl.fit(X_train, y_train)
            y_hat = automl.predict(X_test)
            acc = sklearn.metrics.accuracy_score(y_test, y_hat)
            print(acc)
            with open("./TestingAcc/" + filename[:-4] + "_test_acc" + str(i) + ".txt", 'w') as outfile:
                outfile.write(str(acc))
            y_hat = automl.predict(X_train)
            acc = sklearn.metrics.accuracy_score(y_train, y_hat)
            print(acc)
            with open("./TrainingAcc/" + filename[:-4] + "_train_acc" + str(i) + ".txt", 'w') as outfile:
                outfile.write(str(acc))

            hp = convertTextToConfig(automl.show_models())
            print(hp)
            with open("./ManualTesting/FinalHP/" + filename[:-4] + "_hp_" + str(i) + ".txt", 'w') as outfile:
                outfile.write(str(hp))



def test():
    global D2C, mode, path

    models = []

    with open(modelName, "rb") as fp:
        models = pickle.load(fp)

    for filename in os.listdir(path):
        hps = []
        try:
            meta, hp = separate(catToNum(pd.read_csv(("./MetaConfigData/" + filename[:-4] + "_meta_config_regression.csv"), skipinitialspace=True)))
            print(meta)
            print(hp)
        except Exception as e:
            print(e)
            continue

        print(models)
        for i in usedHPSVM:
            hp_hat = models[i].predict(meta)
            hps.append(hp_hat[0])
            print("Predicted:")
            print(hp_hat)
            np.savetxt("./ManualTesting/PredictedHP/" + filename[:-4] + "_testing_res_" + str(i) + ".txt",
                       hp_hat)
        print(hps)
        default_hp = [1.0, 0.1, 0.0, 1e-3, 3]
        #hps = default_hp
        whp.replace('./autosklearn/pipeline/components/classification/libsvm_svc.py', hps)
        l2 = hps[0]
        rate = hps[1]
        iter = hps[2]
        max_leaf = hps[3]
        min_leaf = hps[4]

        D2C = False
        mode = "classification"
        fit_and_save_over_time(path, filename)
        #fit_custom_and_default(path, filename)




def readTestInput():
    isRead = True
    while isRead:
        try:
            readData()
            if path == loadCommand:
                meta = []
                with open(fileListName, "rb") as fp:
                    global fileList
                    fileList = pickle.load(fp)

            test()
        except Exception as e:
            print("failed to read")
            print(e)

def modifyDataset():
    df = pd.read_csv("./ManualTestDatasets/Kaggle-data.csv")
    del df['md5']
    del df['ID']
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('legitimate')))
    df = df.loc[:, cols]
    df.to_csv("./Kaggle-data-malware-new.csv")

def main():
    #This line for database training/dataset search
    #meta = readDatabaseInput()

    #This line for testing (training on some dataset)
    readTestInput()



main()
