import MetaFeatures.Output
import pandas as pd
import numpy as np

path = ""
fileList = []
searchTerm = ""

def readData():
    folderpath = input("Path to folder of dataset :")
    global path
    path = folderpath

def getMetafeatures():
    return 1

def readInput():
    return 1

def search(term, meta):
    return 1


def main():
    readData()
    meta = getMetafeatures()
    readInput()
    search(searchTerm, meta)

