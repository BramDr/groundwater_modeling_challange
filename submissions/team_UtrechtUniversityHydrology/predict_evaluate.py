import pathlib as pl
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime as dt
import torch.nn.functional as fun
import pickle

from SequenceModel import SequenceModel

## Options
save_dir = pl.Path("./saves")
sub_dir = pl.Path("../team_example")
dir_out = pl.Path("./saves")
team_out = pl.Path("./")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
cuda = True

location = locations[0]
for location in locations:
    print(location)
    
    x_transformer_out = pl.Path("{}/{}/x_transformer.pkl".format(dir_out, location))
    y_transformer_out = pl.Path("{}/{}/y_transformer.pkl".format(dir_out, location))    
    dataset_file = pl.Path("{}/{}/dataset_predict.pt".format(save_dir,
                                                     location))
    state_dict_file = pl.Path("{}/{}/best_state_dict.pt".format(save_dir,
                                                                location))
    sub_file = pl.Path("{}/submission_form_{}.csv".format(sub_dir,
                                                          location))

    dataset = torch.load(dataset_file)
    state_dict = torch.load(state_dict_file)
    sub = pd.read_csv(sub_file)
    with open(x_transformer_out, 'rb') as file:
        x_transformer = pickle.load(file)
    with open(y_transformer_out, 'rb') as file:
        y_transformer = pickle.load(file)

    ## Setup model
    in_size = state_dict["linear_in.weight"].shape[1]
    hidden_size = state_dict["linear_in.weight"].shape[0]
    out_size = state_dict["linear_out.weight"].shape[0]
    n_lstm = len([key for key in state_dict.keys() if "lstm.weight_ih" in key])
    model = SequenceModel(in_size=in_size,
                        hidden_size=hidden_size,
                        out_size=out_size,
                        n_lstm=n_lstm,
                        dropout_rate=0,
                        cuda=cuda)
    model.load_state_dict(state_dict=state_dict)
    model = model.train(mode = False)
    print(model)

    ## Setup predict
    dataset_len = len(dataset)
    dataset_indices = [index for index in range(dataset_len)]

    ## Predict
    x, y_true = dataset[0]
    with torch.no_grad():
        y_pred = model.forward(x)
    
    missing = torch.isnan(y_true)
    loss = fun.mse_loss(input = y_pred[~missing], target = y_true[~missing])
    print(loss)
    
    ## Denormalize
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    y_true_denormalize = np.reshape(a=y_true, newshape=(-1, y_true.shape[2])).copy()
    y_true_denormalize = y_transformer.inverse_transform(X=y_true_denormalize)
    y_true_denormalize = np.reshape(a=y_true_denormalize, newshape=y_true.shape)
    
    y_pred_denormalize = np.reshape(a=y_pred, newshape=(-1, y_pred.shape[2])).copy()
    y_pred_denormalize = y_transformer.inverse_transform(X=y_pred_denormalize)
    y_pred_denormalize = np.reshape(a=y_pred_denormalize, newshape=y_pred.shape)
    
    ## Write    
    y_pred_bounds = np.quantile(a=y_pred_denormalize, q=(0.05, 0.95))
    
    sequences = sub[sub.keys()[0]].to_list()
    sequences.sort()
    sequences = [dt.datetime.strptime(sequence, "%Y-%m-%d") for sequence in sequences]
    
    y_true_extended = [y_true_denormalize[0, dataset.sequences.index(t), 0] for t in sequences]
    y_pred_extended = [y_pred_denormalize[0, dataset.sequences.index(t), 0] for t in sequences]
    
    plt.plot(sequences, y_true_extended)
    plt.plot(sequences, y_pred_extended)
    plt.show()
    
    sub["Simulated Head"] = y_pred_extended
    sub["95% Lower Bound"] = y_pred_bounds[0]
    sub["95% Upper Bound"] = y_pred_bounds[1]
    
    sub_out = pl.Path("{}/submission_form_{}.csv".format(team_out,
                                                          location))
    #sub.to_csv(sub_out)