import pathlib as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import datetime as dt
import torch
import pickle

from SequenceDataset import SequenceDataset

## Options
data_dir = pl.Path("../../data")
save_dir = pl.Path("./saves")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
dir_out = pl.Path("./saves")
cuda = True

## Load data
location = locations[0]
location = "Sweden_1"
for location in locations:
    print(location)
    
    input_file = pl.Path("{}/{}/input.npy".format(save_dir, location))
    output_file = pl.Path("{}/{}/output.npy".format(save_dir, location))
    dates_file = pl.Path("{}/{}/dates.csv".format(save_dir, location))
    
    input = np.load(file=input_file)
    output = np.load(file=output_file)
    dates = pd.read_csv(dates_file, index_col=0)
    
    active_sel = ~np.isnan(output).flatten()
    active_indices = np.where(active_sel)[0]
    split_index = int(len(active_indices) * .6)
    
    train_dates = dates.iloc[active_indices[:split_index]]
    train_x = input[:,active_indices[:split_index],:]
    train_y = output[:,active_indices[:split_index],:]
    
    test_dates = dates.iloc[active_indices[split_index:]]
    test_x = input[:,active_indices[split_index:],:]
    test_y = output[:,active_indices[split_index:],:]
    
    ## Create dataset
    train_dataset = SequenceDataset(x = train_x,
                                    y = train_y,
                                    dates = pd.to_datetime(train_dates["0"].values),
                                    cuda=cuda)
    
    test_dataset = SequenceDataset(x = test_x,
                                    y = test_y,
                                    dates = pd.to_datetime(test_dates["0"].values),
                                    cuda=cuda)
    
    train_out = pl.Path("{}/{}/train_dataset.pt".format(dir_out, location))    
    train_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_dataset, train_out)
    
    test_out = pl.Path("{}/{}/test_dataset.pt".format(dir_out, location))    
    test_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_dataset, test_out)
