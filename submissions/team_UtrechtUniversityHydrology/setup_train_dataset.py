import pathlib as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from SequenceDataset import SequenceDataset

## Options
data_dir = pl.Path("../../data")
dir_out = pl.Path("./saves")
normalize_x_out = pl.Path("./saves/normalize_x.npy")
normalize_y_out = pl.Path("./saves/normalize_y.npy")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
cuda = True

## Load data
for location in locations:
    print(location)
    
    input_file = pl.Path("{}/{}/input_data.csv".format(data_dir,
                                                    location))
    output_file = pl.Path("{}/{}/heads.csv".format(data_dir,
                                                location))

    input = pd.read_csv(input_file)
    output = pd.read_csv(output_file)
    
    in_features = input.keys().to_list()[1:]
    out_features = output.keys().to_list()[1:]
    
    sequences = list(set(output[output.keys()[0]].to_list()))
    overlapping = [time in sequences for time in input[input.keys()[0]]]
    
    x = input.loc[overlapping, in_features].to_numpy()
    x = np.expand_dims(a = x, axis = 0)
    y = output.loc[:, out_features].to_numpy()
    y = np.expand_dims(a = y, axis = 0)
    
    ## Normalize data
    x_min = np.min(a = x, axis = 1, keepdims=True)
    x_max = np.max(a = x, axis = 1, keepdims=True)
    x_diff = x_max - x_min

    y_min = np.min(a = y, axis = 1, keepdims=True)
    y_max = np.max(a = y, axis = 1, keepdims=True)
    y_diff = y_max - y_min

    x_norm = (x - x_min) / x_diff
    y_norm = (y - y_min) / y_diff
    
    ## Create dataset
    dataset = SequenceDataset(x = x_norm,
                            y = y_norm,
                            samples=locations,
                            sequences=sequences,
                            in_features=in_features,
                            out_features=out_features,
                            cuda=cuda)
    
    dataset_out = pl.Path("{}/{}/dataset.pt".format(dir_out, location))
    normalize_x_out = pl.Path("{}/{}/normalize_x.npy".format(dir_out, location))
    normalize_y_out = pl.Path("{}/{}/normalize_y.npy".format(dir_out, location))
    
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, dataset_out)
    normalize_x_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(normalize_x_out, (x_min, x_max))
    normalize_y_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(normalize_y_out, (y_min, y_max))

    ## Visual check
    plt.plot(y.flatten())
    plt.plot(output["head"])
    plt.show()
