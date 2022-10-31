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
normalize_x_out = pl.Path("./saves/normalize_x.npy")
normalize_y_out = pl.Path("./saves/normalize_y.npy")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
cuda = True

## Load data
location = locations[0]
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
    
    x_sequences = list(set(input[input.keys()[0]].to_list()))
    x_sequences.sort()
    x_sequences = [dt.datetime.strptime(sequence, "%Y-%m-%d") for sequence in x_sequences]
    
    y_sequences = list(set(output[output.keys()[0]].to_list()))
    y_sequences.sort()
    y_sequences = [dt.datetime.strptime(sequence, "%Y-%m-%d") for sequence in y_sequences]
    
    overlapping = [t in y_sequences for t in x_sequences]
    sequences = y_sequences
    
    x = input.loc[overlapping, in_features].to_numpy()
    x = np.expand_dims(a = x, axis = 0)
    y = output.loc[:, out_features].to_numpy()
    y = np.expand_dims(a = y, axis = 0)

    ## Visual check
    plt.plot(sequences, x[:,:,0].flatten())
    plt.plot(sequences, input.loc[overlapping, in_features[0]])
    plt.show()
    
    plt.plot(sequences, y.flatten())
    plt.plot(sequences, output[out_features[0]])
    plt.show()
    
    ## Normalize data
    #x_transformer = pp.QuantileTransformer(n_quantiles=10000)
    x_transformer = pp.MinMaxScaler()
    x_norm = np.reshape(a=x, newshape=(-1, x.shape[2])).copy()
    x_norm = x_transformer.fit_transform(X=x_norm)
    x_norm = np.reshape(a=x_norm, newshape=x.shape)
    
    #y_transformer = pp.QuantileTransformer(n_quantiles=10000)
    y_transformer = pp.MinMaxScaler()
    y_norm = np.reshape(a=y, newshape=(-1, y.shape[2])).copy()
    y_norm = y_transformer.fit_transform(X=y_norm)
    y_norm = np.reshape(a=y_norm, newshape=y.shape)
    
    ## Create dataset
    dataset = SequenceDataset(x = x_norm,
                              y = y_norm,
                              samples=locations,
                              sequences=sequences,
                              in_features=in_features,
                              out_features=out_features,
                              cuda=cuda)
    
    dataset_out = pl.Path("{}/{}/dataset.pt".format(dir_out, location))    
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, dataset_out)
    
    x_transformer_out = pl.Path("{}/{}/x_transformer.pkl".format(dir_out, location))
    y_transformer_out = pl.Path("{}/{}/y_transformer.pkl".format(dir_out, location))
    x_transformer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(x_transformer_out, 'wb') as file:
        pickle.dump(x_transformer, file)
    y_transformer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(y_transformer_out, 'wb') as file:
        pickle.dump(y_transformer, file)
