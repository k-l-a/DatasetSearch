import openml as oml
import os
import numpy as np
import pandas as pd


def main():
    datasets = oml.datasets.list_datasets(output_format='dataframe')
    datasets.to_csv('OpenMlDatasetCatalog')
    for id in datasets['did']:
        dataset = oml.datasets.get_dataset(id)
        df, y, cat, cols = dataset.get_data(target=None, dataset_format='dataframe')
        if len(df) > 10000:
            continue
        tag = dataset.tag
        postfixes = np.vectorize(lambda x: '_cat' if x else '_num')(cat).tolist()
        assert len(cols) == len(postfixes)
        df.columns = [cols[i] + postfixes[i] for i in range(len(cols))]
        file_name = dataset.name + "_" + tag + "_" + str(id) + ".csv"
        file_name = ''.join(c for c in file_name if c not in '<>:"|\/?*' )
        df.to_csv(os.path.join('OpenMlDatasets', file_name))


def get_rest(starting_id=1):
    df = pd.read_csv("OpenMlDatasetCatalog")
    existing_files = os.listdir("OpenMlDatasets")
    existing_ids = [int(i.split("_")[-1].split('.')[0], 10) for i in existing_files]
    all_ids = df['did']
    deprecated_ids = [202, 386]
    rest_ids = [i for i in all_ids if i >= starting_id and i not in existing_ids and i not in deprecated_ids]
    for id in rest_ids:
        print(id)
        dataset = oml.datasets.get_dataset(id)
        df, y, cat, cols = dataset.get_data(target=None, dataset_format='dataframe')
        if len(df) > 10000:
            continue
        tag = dataset.tag
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
    get_rest(starting_id=388)
