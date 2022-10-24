import pathlib as pl
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as fun

from SequenceModel import SequenceModel

## Options
save_dir = pl.Path("./saves")
sub_dir = pl.Path("../team_example")
dir_out = pl.Path("./saves")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
cuda = True

location = locations[0]
for location in locations:
    print(location)
    
    dataset_file = pl.Path("{}/{}/dataset_predict.pt".format(save_dir,
                                                     location))
    state_dict_file = pl.Path("{}/{}/best_state_dict.pt".format(save_dir,
                                                                location))
    sub_file = pl.Path("{}/submission_form_{}.csv".format(sub_dir,
                                                location))

    ## Load
    dataset = torch.load(dataset_file)
    state_dict = torch.load(state_dict_file)
    sub = pd.read_csv(sub_file)

    ## Setup model
    in_size = state_dict["linear_in.weight"].shape[1]
    hidden_size = state_dict["linear_in.weight"].shape[0]
    out_size = state_dict["linear_out.weight"].shape[0]
    model = SequenceModel(in_size=in_size,
                        hidden_size=hidden_size,
                        out_size=out_size,
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
        
    #y_true = y_true.detach().cpu().numpy()
    #y_pred = y_pred.detach().cpu().numpy()
    
    missing = torch.isnan(y_true)
    loss = fun.mse_loss(input = y_pred[~missing], target = y_true[~missing])
    print(loss)
    
    ## Visual check
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    plt.plot(y_true.flatten())
    plt.plot(y_pred.flatten())
    plt.show()
    
    ## Write
    y_pred_bounds = np.quantile(a=y_pred, q=(0.05, 0.95))
    sub["Simulated Head"] = y_pred.flatten()
    sub["95% Lower Bound"] = y_pred_bounds[0]
    sub["95% Upper Bound"] = y_pred_bounds[1]