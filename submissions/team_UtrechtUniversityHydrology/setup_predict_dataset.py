import pathlib as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from SequenceDataset import SequenceDataset

## Options
data_dir = pl.Path("../../data")
sub_dir = pl.Path("../team_example")
dir_out = pl.Path("./saves")
normalize_x_out = pl.Path("./saves/normalize_x.npy")
normalize_y_out = pl.Path("./saves/normalize_y.npy")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
cuda = True

## Load data
location = locations[0]
for location in locations:
    print(location)
    
    normalize_x_out = pl.Path("{}/{}/normalize_x.npy".format(dir_out, location))
    normalize_y_out = pl.Path("{}/{}/normalize_y.npy".format(dir_out, location))
    x_min, x_max = np.load(normalize_x_out)
    y_min, y_max = np.load(normalize_y_out)
    x_diff = x_max - x_min
    y_diff = y_max - y_min
    
    input_file = pl.Path("{}/{}/input_data.csv".format(data_dir,
                                                    location))
    output_file = pl.Path("{}/{}/heads.csv".format(data_dir,
                                                location))
    sub_file = pl.Path("{}/submission_form_{}.csv".format(sub_dir,
                                                location))

    input = pd.read_csv(input_file)
    output = pd.read_csv(output_file)
    sub = pd.read_csv(sub_file)
    
    in_features = input.keys().to_list()[1:]
    out_features = output.keys().to_list()[1:]
    
    sequences = list(set(sub[sub.keys()[0]].to_list()))
    sequences.sort()
    x_overlap = [t in sequences for t in input[input.keys()[0]].to_list()]
    y_overlap = [t in output[output.keys()[0]].to_list() for t in sequences]
    
    x = input.loc[x_overlap, in_features].to_numpy()
    x = np.expand_dims(a = x, axis = 0)
    y = output.loc[:, out_features].to_numpy()
    y = np.expand_dims(a = y, axis = 0)
    
    y_extended = np.full((y.shape[0], x.shape[1], y.shape[2]), fill_value=np.nan)
    y_extended[:, y_overlap, :] = y
    
    ## Normalize data
    x_norm = (x - x_min) / x_diff
    y_norm = (y_extended - y_min) / y_diff
    
    ## Create dataset
    dataset = SequenceDataset(x = x_norm,
                              y = y_norm,
                              samples=locations,
                              sequences=sequences,
                              in_features=in_features,
                              out_features=out_features,
                              cuda=cuda)
    
    dataset_out = pl.Path("{}/{}/dataset_predict.pt".format(dir_out, location))
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, dataset_out)

    ## Visual check
    plt.plot(y.flatten())
    plt.plot(output["head"])
    plt.show()
