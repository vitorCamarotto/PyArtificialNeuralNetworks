# normalize class

import pandas as pd
import os
import numpy as np

print('normalizing data...')

def read_csv_adjusted_delimiter(file_path, delimiter, decimal):
    return pd.read_csv(file_path, delimiter=delimiter, decimal=decimal)

def normalize_data(data, method='range'):
    if method == 'range':
        mean_value = data.mean()
        std_value = data.std()
        normalized_data = (data - mean_value) / std_value
        norm_details = {'mean': mean_value, 'std': std_value}
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized_data, norm_details


def save_normalized_data(paths):
    cnn_csv_data = read_csv_adjusted_delimiter(paths['TrainingPathCNN'], delimiter=',', decimal='.').iloc[1:, 2:].values
    xCNN, norm_detailsCNN = normalize_data(cnn_csv_data)

    with open(os.path.join(paths['DestinationFolder'], 'norm_detailsCNN.npy'), 'wb') as f:
        np.save(f, norm_detailsCNN)

    mlp_csv_data = read_csv_adjusted_delimiter(paths['TrainingPathMLP'], ';', ',').iloc[1:, 2:].values
    xMLP, norm_detailsMLP = normalize_data(mlp_csv_data)

    with open(os.path.join(paths['DestinationFolder'], 'norm_detailsMLP.npy'), 'wb') as f:
        np.save(f, norm_detailsMLP)

    target_csv_data = read_csv_adjusted_delimiter(paths['TargetPath'],  ';', ',').iloc[1:, 2:3].values
    target, norm_detailsTarget = normalize_data(target_csv_data)

    with open(os.path.join(paths['folderOfResults'], 'norm_detailsTargets.npy'), 'wb') as f:
        np.save(f, norm_detailsTarget)

    return xCNN, xMLP, target
