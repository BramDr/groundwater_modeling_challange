import pathlib as pl
import random
import torch
import torch.optim as opt
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import torch.nn.functional as fun
import pickle

from SequenceModel import SequenceModel

## Options
save_dir = pl.Path("./saves")
dir_out = pl.Path("./saves")
locations = ["Germany", "Netherlands", "Sweden_1", "Sweden_2", "USA"]
epochs = 1000
hidden_size = 512
sequence_size = 100
dropout_rate = 0.25
learning_rate = 1e-2
transformer_base = pp.QuantileTransformer
seed = 19920223
cuda = True

## Load
location = locations[0]
for location in locations:
    print(location)
    
    train_file = pl.Path("{}/{}/train_dataset.pt".format(save_dir, location))    
    test_file = pl.Path("{}/{}/test_dataset.pt".format(save_dir, location))
    
    train_dataset = torch.load(train_file)
    test_dataset = torch.load(test_file)
    
    train_dataset.set_sequence_size(sequence_size)
    x_transformer, y_transformer = train_dataset.transform(x_transformer = transformer_base(), 
                                                           y_transformer = transformer_base(),
                                                           fit=True)
    x_transformer, y_transformer = test_dataset.transform(x_transformer = x_transformer, 
                                                            y_transformer = y_transformer,
                                                            fit=False)

    x_transformer_out = pl.Path("{}/{}/x_transformer.pkl".format(dir_out, location))
    x_transformer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(file=x_transformer_out, mode="wb") as file:
        pickle.dump(x_transformer, file)
    
    y_transformer_out = pl.Path("{}/{}/y_transformer.pkl".format(dir_out, location))
    y_transformer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(file=y_transformer_out, mode="wb") as file:
        pickle.dump(y_transformer, file)

    ## Setup model
    in_size = train_dataset.x.shape[2]
    out_size = train_dataset.y.shape[2]
    model = SequenceModel(in_size=in_size,
                        hidden_size=hidden_size,
                        out_size=out_size,
                        dropout_rate=dropout_rate,
                        cuda=cuda)
    print(model)

    # Setup training
    optimizer = opt.Adam(params = model.parameters(), 
                         lr=learning_rate)

    ## Train and save
    random.seed(seed)
    
    best_loss = float("inf")
    for epoch in range(epochs):
        
        # TRAINING
        x, y_true = train_dataset[0]
        model = model.train(mode = True)
        y_pred = model.forward(x)
        
        true_sel = ~torch.isnan(y_true)
        train_loss = fun.mse_loss(input = y_pred[true_sel], target = y_true[true_sel])
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # TESTING
        x, y_true = test_dataset[0]
        model = model.train(mode = False)
        with torch.inference_mode():
            y_pred = model.forward(x)
            
        true_sel = ~torch.isnan(y_true)
        test_loss = fun.mse_loss(input = y_pred[true_sel], target = y_true[true_sel])
        
        print("Epoch {}: train loss {}, test loss {}".format(epoch, train_loss, test_loss))
        
        if test_loss < best_loss:
            best_loss = test_loss
            state_dict_out = pl.Path("{}/{}/best_state_dict.pt".format(dir_out,
                                                                       location))
            torch.save(model.state_dict(), state_dict_out)
    
    ## Visual check
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    plt.scatter(test_dataset.dates, y_true.flatten(), color = '#88c999')
    plt.plot(test_dataset.dates, y_pred.flatten())
    plt.show()
