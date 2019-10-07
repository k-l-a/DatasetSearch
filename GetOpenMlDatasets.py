import openml as oml
import os
import numpy as np
import pandas as pd


def main():
    datasets = oml.datasets.list_datasets(output_format='dataframe')
    datasets.to_csv('OpenMlDatasetCatalog')
    for id in datasets['did']:
        print(id)
        print(id)
        dataset = oml.datasets.get_dataset(id)
        df, y, cat, cols = dataset.get_data(target=None, dataset_format='dataframe')
        postfixes = np.vectorize(lambda x: '_cat' if x else '_num')(cat).tolist()
        assert len(cols) == len(postfixes)
        df.columns = [cols[i] + postfixes[i] for i in range(len(cols))]
        file_name = dataset.name + "_" + str(id) + ".csv"
        file_name = ''.join(c for c in file_name if c not in '<>:"|\/?*' )
        df.to_csv(os.path.join('OpenMlDatasets', file_name))


def get_rest():
    df = pd.read_csv("OpenMlDatasetCatalog")
    existing_files = os.listdir("OpenMlDatasets")
    existing_ids = [int(i.split("_")[-1].split('.')[0], 10) for i in existing_files]
    all_ids = df['did']
    rest_ids = [i for i in all_ids if i not in existing_ids]
    print(rest_ids[:10])
    for id in rest_ids:
        print(id)
        dataset = oml.datasets.get_dataset(id)
        df, y, cat, cols = dataset.get_data(target=None, dataset_format='dataframe')
        postfixes = np.vectorize(lambda x: '_cat' if x else '_num')(cat).tolist()
        assert len(cols) == len(postfixes)
        df.columns = [cols[i] + postfixes[i] for i in range(len(cols))]
        file_name = dataset.name + "_" + str(id) + ".csv"
        file_name = ''.join(c for c in file_name if c not in '<>:"|\/?*' )
        df.to_csv(os.path.join('OpenMlDatasets', file_name))


def refresh_catalog():
    datasets = oml.datasets.list_datasets(output_format='dataframe')
    datasets.to_csv('OpenMlDatasetCatalog')

if __name__ =="__main__":
    main()
