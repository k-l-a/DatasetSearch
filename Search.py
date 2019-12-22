import numpy as np
import autosklearn.classification
import warnings
import faiss
from MetaFeatures.Output import MetaFeatures
from MetaFeatures import Output
from dataset2vec import main as d2v
from dataset2vec import config
from enum import Enum
import ConvertCsvtoSvmLight
import math
import os
import pandas as pd
import pickle

warnings.filterwarnings("ignore", "Mean of empty slice")

path = ""
defaultPath = "./SavedDatasets/"
savePath = "./SavedDatasets/"
exitCommand = "exit"
loadCommand = "load"
saveCommand = "save"
indexFilename = "./saved_index2"
fileListName = "./saved_filelist2"
indexType = "L2"
fileList = []
searchTerm = ""
weights = []

useDefault = False
useCSV = True
used2v = False
fromCsv = False
isWeighted = False

datasetProcessD2C = False


d = 50
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
            continue

    arr = np.array(featureList).astype('float32')
    print(arr.shape)
    return arr


# Function to process dataset. Accepts a dataset (x, y) and returns a 'dataset'(x', y') that has been processed
def processDataset(x, y, model):
    if datasetProcessD2C:
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
    if datasetProcessD2C:
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
            x, y = separate(catToNum(pd.read_csv((termPath))).fillna(0))
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
            else:
                meta = getMetafeatures()
            isRead = False
        except Exception as e:
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


def main():
    meta = readDatabaseInput()
    while(readSearchInput()):
        try:
            search(searchTerm, meta, path == loadCommand)
        except Exception as e:
            print(e)

main()
