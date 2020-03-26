# DatasetSearch

Dataset search engine prototype

Instructions:
1. Run Search.py with D2C = false
2. Enter datasets database directory (e.g. ./ConvertedDatasets/)
3. Enter filepath of dataset to be used as search term (e.g. ./ConvertedDatasets/somefilename.csv)
4. Filename of closest dataset to the search term from the database will be printed.


Dataset search is done using faiss (https://github.com/facebookresearch/faiss) and metafeatures extraction is done using https://github.com/Seris370/Test


Hyperparameter Prediction

Instructions:
This is separated into 2 parts:
    Hyperparameter Training:
    1. Change appropriate variables in Search.py accordingly (depending on # of hyperparam, etc.)
    2. Run Search.py with readDatabaseInput
    3. Enter folder directory containing the metafeatures + hyperparam dataset
    4. Wait for the training to finish.
    5. Model will be saved on the current directory based on model name variable

    Prediction:
    1. Run Search,py with readTestInput
    2. Enter folder directory containing the datasets to train on
    3. Wait for the training to finish.
    4. Accuracy of the resulting model will be saved. (default: TestingAcc or TrainingAcc folder)
