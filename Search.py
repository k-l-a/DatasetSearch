import numpy as np
import faiss
from MetaFeatures import Output
import os

path = ""
defaultPath = "./ConvertedDatasets/"
fileList = []
searchTerm = ""

useDefault = False
d = 50
k = 5

def readData():
    folderpath = input("Path to dataset base folder :")
    global path
    if useDefault:
        path = defaultPath
    else:
        path = folderpath

def getMetafeatures():
    featureList = []
    global d
    for filename in os.listdir(path):
        print(filename)
        metaFeatures = normalizeDimension(Output.get_metafeatures(path + filename))
        featureList.append(metaFeatures)
        fileList.append(filename)
    return np.array(featureList).astype('float32')

def normalizeDimension(vector):
    if len(vector) < d:
        for i in range(d - len(vector)):
            vector.append(0)  # Pad the features to reach desired dimension
    else:
        vector = vector[:d]
    return vector

def readSearchTerm():
    term = input("Dataset to use as search term :")
    global searchTerm
    searchTerm = term

def search(termPath, metaFeatureList):
    global d, k
    index = faiss.IndexFlatL2(d)
    print(index.is_trained)
    index.add(metaFeatureList)
    termFeatures = normalizeDimension(Output.get_metafeatures(termPath))
    termFeatures = np.array([termFeatures]).astype('float32')
    D, I = index.search(termFeatures, k)
    print(I[:5])
    for i in I[:5]:
        for j in i:
            print(fileList[j])

def main():
    readData()
    meta = getMetafeatures()
    readSearchTerm()
    search(searchTerm, meta)


main()