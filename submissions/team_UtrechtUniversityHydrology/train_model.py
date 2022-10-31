import pathlib as pl
import random
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.nn.functional as fun

from SequenceModel import SequenceModel

## Options
save_dir = pl.Path("./saves")
dir_out = pl.Path("./saves")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
epochs = 5000
hidden_size = 512
n_lstm = 3
dropout_rate = 0.25
seed = 19920223
cuda = True

## Load
location = locations[0]
for location in locations:
    print(location)
    
    dataset_file = pl.Path("{}/{}/dataset.pt".format(save_dir,
                                                     location))
    
    dataset = torch.load(dataset_file)

    ## Setup model
    in_size = len(dataset.in_features)
    out_size = len(dataset.out_features)
    model = SequenceModel(in_size=in_size,
                        hidden_size=hidden_size,
                        n_lstm=n_lstm,
                        out_size=out_size,
                        dropout_rate=dropout_rate,
                        cuda=cuda)
    model = model.train(mode = True)
    print(model)

    # Setup training
    optimizer = opt.Adam(params = model.parameters())

    dataset_len = len(dataset)
    dataset_indices = [index for index in range(dataset_len)]

    ## Train and save
    random.seed(seed)
    
    best_loss = float("inf")
    for epoch in range(epochs):
        indices = random.sample(population=dataset_indices,
                                k = dataset_len)
        
        epoch_loss = 0
        for index in indices:
            x, y_true = dataset[index]
            y_pred = model.forward(x)
            
            loss = fun.mse_loss(input = y_pred, target = y_true)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(indices)
        print("Epoch {}: {}".format(epoch, epoch_loss))
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state_dict_out = pl.Path("{}/{}/best_state_dict.pt".format(dir_out,
                                                                       location))
            torch.save(model.state_dict(), state_dict_out)

    ## Visual check
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    plt.plot(dataset.sequences, y_true.flatten())
    plt.plot(dataset.sequences, y_pred.flatten())
    plt.show()
