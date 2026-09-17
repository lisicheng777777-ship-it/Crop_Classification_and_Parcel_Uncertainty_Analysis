import sys
import math
import random
import itertools
import csv
import numpy as np
import pandas as pd


def _clean_feature_frame(X_data, source_name):
    values = X_data.apply(pd.to_numeric, errors="coerce")
    arr = values.to_numpy(dtype=np.float32)
    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return arr

    medians = values.replace([np.inf, -np.inf], np.nan).median(axis=0, skipna=True)
    medians = medians.fillna(0.0).to_numpy(dtype=np.float32)
    row_idx, col_idx = np.where(invalid_mask)
    arr[row_idx, col_idx] = medians[col_idx]
    print(f"Warning: replaced {invalid_count} invalid feature values in {source_name} with column medians.")
    return arr


def readSITSData(name_file):
    """
        Read the data contained in name_file
        INPUT:
            - name_file: file where to read the data
        OUTPUT:
            - X: variable vectors for each example
            - polygon_ids: id polygon (use e.g. for validation set)
            - Y: label for each example
    """

    data = pd.read_csv(name_file, sep=',', header=None)

    y_data = data.iloc[:, 0]
    y = np.asarray(y_data.values, dtype='uint8')

    polygonID_data = data.iloc[:, 1]
    polygon_ids = polygonID_data.values
    polygon_ids = np.asarray(polygon_ids, dtype='uint64')

    X_data = data.iloc[:, 2:]
    X = _clean_feature_frame(X_data, name_file)

    return X, polygon_ids, y


def read_minMaxVal(minmax_file):
    with open(minmax_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError(f"Normalization file must contain min and max rows: {minmax_file}")
    min_per = rows[0]
    max_per = rows[1]
    min_per = [float(k) for k in min_per]
    min_per = np.array(min_per)
    max_per = [float(k) for k in max_per]
    max_per = np.array(max_per)
    return min_per, max_per


def save_minMaxVal(minmax_file, min_per, max_per):
    with open(minmax_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(min_per)
        writer.writerow(max_per)


def extractValSet(X_train, polygon_ids_train, y_train, val_rate=0.2,
                  return_polygon_ids=False, sample_weights=None):
    unique_pol_ids_train, indices = np.unique(polygon_ids_train, return_inverse=True)  # -- pold_ids_train = unique_pol_ids_train[indices]
    nb_pols = len(unique_pol_ids_train)
    ind_shuffle = list(range(nb_pols))
    random.shuffle(ind_shuffle)
    list_indices = [[] for i in range(nb_pols)]
    shuffle_indices = [[] for i in range(nb_pols)]
    [list_indices[ind_shuffle[val]].append(idx) for idx, val in enumerate(indices)]

    final_ind = list(itertools.chain.from_iterable(list_indices))
    m = len(final_ind)
    final_train = int(math.ceil(m * (1.0 - val_rate)))
    shuffle_polygon_ids_train = polygon_ids_train[final_ind]
    id_final_train = shuffle_polygon_ids_train[final_train]

    while shuffle_polygon_ids_train[final_train - 1] == id_final_train:
        final_train = final_train - 1

    new_X_train = X_train[final_ind[:final_train], :, :]
    new_y_train = y_train[final_ind[:final_train]]
    new_polygon_ids_train = polygon_ids_train[final_ind[:final_train]]
    new_X_val = X_train[final_ind[final_train:], :, :]
    new_y_val = y_train[final_ind[final_train:]]
    new_polygon_ids_val = polygon_ids_train[final_ind[final_train:]]

    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if len(sample_weights) != len(X_train):
            raise ValueError("sample_weights must be row-aligned with X_train")
        new_weights_train = sample_weights[final_ind[:final_train]]
        new_weights_val = sample_weights[final_ind[final_train:]]
        if return_polygon_ids:
            return (new_X_train, new_polygon_ids_train, new_y_train, new_weights_train,
                    new_X_val, new_polygon_ids_val, new_y_val, new_weights_val)
        return new_X_train, new_y_train, new_weights_train, new_X_val, new_y_val, new_weights_val

    if return_polygon_ids:
        return new_X_train, new_polygon_ids_train, new_y_train, new_X_val, new_polygon_ids_val, new_y_val
    return new_X_train, new_y_train, new_X_val, new_y_val

# This code was implemented from http://devseed.com/sat-ml-training/Randomforest_cropmapping-with_GEE#Model-training

