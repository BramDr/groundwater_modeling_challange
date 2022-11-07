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
dir_out = pl.Path("./saves")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
cuda = True

## Load data
location = locations[0]
location = "Sweden_1"
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
    
    dates_input = pd.to_datetime(input[input.keys()[0]]).values
    dates_output = pd.to_datetime(output[output.keys()[0]]).values
    dates = np.concatenate((dates_input, dates_output))
    date_min = min(dates)
    date_max = max(dates)
    dates_full = pd.date_range(start = date_min, end = date_max)
    
    input_array = input.loc[:, in_features].to_numpy()
    samples_len = 1
    dates_len = len(dates_full)
    features_len = len(in_features)
    input_array_full = np.full((samples_len, dates_len, features_len), fill_value=np.nan, dtype=np.float32)
    
    dates_selected = np.array([datum in dates_input for datum in dates_full])
    dates_indices = np.where(dates_selected)[0]
    for from_index, to_index in enumerate(dates_indices):
        input_array_full[:, to_index, :] = input_array[from_index, :]
        
    output_array = output.loc[:, out_features].to_numpy()
    samples_len = 1
    dates_len = len(dates_full)
    features_len = len(out_features)
    output_array_full = np.full((samples_len, dates_len, features_len), fill_value=np.nan, dtype=np.float32)
    
    dates_selected = np.array([datum in dates_output for datum in dates_full])
    dates_indices = np.where(dates_selected)[0]
    for from_index, to_index in enumerate(dates_indices):
        output_array_full[:, to_index, :] = output_array[from_index, :]
    
    input_out = pl.Path("{}/{}/input.npy".format(dir_out, location))    
    input_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(file=input_out, arr = input_array_full)
    
    output_out = pl.Path("{}/{}/output.npy".format(dir_out, location))    
    output_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(file=output_out, arr = output_array_full)
    
    dates_out = pl.Path("{}/{}/dates.csv".format(dir_out, location))    
    dates_out.parent.mkdir(parents=True, exist_ok=True)
    dates_full.to_frame().to_csv(dates_out)
    