import openml as oml
import os
import math
import numpy as np
import pandas as pd
from openml.exceptions import OpenMLServerException


def main():
    datasets = refresh_catalog()
    for id in datasets['did']:
        try:
            dataset = oml.datasets.get_dataset(id)
        except OpenMLServerException:
            print("deprecated dataset!")
            continue
        df, y, cat, cols = dataset.get_data(target=None, dataset_format='dataframe')
        if datasets[df['did'] == id]['NumberOfInstances'].sum() > 10000:
            continue
        tag = dataset.tag
        postfixes = np.vectorize(lambda x: '_cat' if x else '_num')(cat).tolist()
        assert len(cols) == len(postfixes)
        df.columns = [cols[i] + postfixes[i] for i in range(len(cols))]
        if y is None:
            cols = df.columns[-1:] + df.columns[:-1]
            df = df[cols]
        else:
            df['class'] = y
        file_name = dataset.name + "_" + str(id) + ".csv"
        file_name = ''.join(c for c in file_name if c not in '<>:"|\/?*' )
        df.to_csv(os.path.join('OpenMlDatasets', file_name))


def get_rest(starting_id=1, threshold=None, criterion=None, show_messages=False, split_frac=None, split_num=None):
    """
    Downloads remaining OpenML datasets from the dataset catalog.
    :param starting_id: OpenMl dataset id from which downloading starts
    :param threshold: number of instances of a dataset, defaults to None
    :param criterion: 'geq' to download datasets with more than threshold instances.
        'leq' to download datasets with less than threshold instances, defaults to None
    :param show_messages: determines whether to show dataset id, defaults to False
    :param split_frac: fraction of columns to split, defaults to None.
    :param split_num: number of columns to split, defaults to None. If both split_frac and split_num are not None,
        choose the method so the number of split columns in sub-datasets is larger.
    :return: void
    """
    datasets = pd.read_csv("OpenMlDatasetCatalog")
    existing_files = os.listdir("OpenMlDatasets")
    existing_ids = [int(i.split("_")[-1].split('.')[0], 10) for i in existing_files]
    all_ids = datasets['did']
    deprecated_ids = []
    rest_ids = [i for i in all_ids if i >= starting_id and i not in existing_ids and i not in deprecated_ids]
    for id in rest_ids:
        if show_messages:
            print(id)
        try:
            dataset = oml.datasets.get_dataset(id)
        except OpenMLServerException:
            if show_messages:
                print("deprecated dataset!")
            continue
        df, y, cat, cols = dataset.get_data(target=None, dataset_format='dataframe')
        if criterion == 'leq':
            assert threshold is not None, "Threshold missing!"
            if datasets[datasets.did == id].NumberOfInstances.sum() > 10000:
                if show_messages:
                    print(id, 'more than', str(threshold), 'instances!')
                continue
        elif criterion == 'geq':
            assert threshold is not None, "Threshold missing!"
            if datasets[datasets.did == id].NumberOfInstances.sum() > 10000:
                if show_messages:
                    print(id, 'more than', str(threshold), 'instances!')
            else:
                continue
        else:
            pass
        postfixes = np.vectorize(lambda x: '_cat' if x else '_num')(cat).tolist()
        assert len(cols) == len(postfixes)
        df.columns = [cols[i] + postfixes[i] for i in range(len(cols))]
        if y is None:
            cols = df.columns
            cols = cols[-1:].append(cols[:-1])
            df = df[cols]
        else:
            df['class'] = y
        file_name = dataset.name + "_" + str(id) + '_0' + ".csv"
        file_name = ''.join(c for c in file_name if c not in '<>:"|\/?*' )
        df.to_csv(os.path.join('OpenMlDatasets', file_name))
        if split_frac is not None and split_num is not None:
            if len(df.columns) * split_frac < split_num:
                split_frac = None
            else:
                split_num = None
        if split_frac is not None:
            num_of_subsets = math.ceil(1 / split_frac)
            for i in range(1, num_of_subsets + 1):
                subset = df.sample(frac=split_frac, random_state=i, axis=1)
                name = dataset.name + "_" + str(id) + "_" + str(i) + ".csv"
                name = ''.join(c for c in name if c not in '<>:"|\/?*')
                subset.to_csv(os.path.join('OpenMlDatasets', name))
        if split_num is not None:
            num_of_subsets = math.ceil(len(df.columns) / split_num)
            for i in range(1, num_of_subsets + 1):
                subset = df.sample(n=split_num, random_state=i, axis=1)
                name = dataset.name + "_" + str(id) + "_" + str(i) + ".csv"
                name = ''.join(c for c in name if c not in '<>:"|\/?*')
                subset.to_csv(os.path.join('OpenMlDatasets', name))


def refresh_catalog():
    datasets = oml.datasets.list_datasets(output_format='dataframe')
    datasets.to_csv('OpenMlDatasetCatalog')
    return datasets


def add_label_to_catalog():
    datasets = pd.read_csv("OpenMlDatasetCatalog")
    #TODO: retrieve labels




if __name__ =="__main__":
    get_rest(starting_id=1, threshold=10000, criterion='leq', show_messages=True, split_frac=0.5, split_num=10)
