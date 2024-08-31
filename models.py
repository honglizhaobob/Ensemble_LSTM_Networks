# Main file defining the various LSTM models as well as helper functions for training
# and creating dataset

import pandas as pd
import numpy as np
import pickle
import scipy
import scipy.stats
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def split_timeseries(timeseries, train_ratio=0.6):
    """
        Given time series stored in an array, 
        partition it into a train and test set 
        with train set accounting for a specified 
        ratio of total data.
    """
    # train-test split for time series
    train_size = int(len(timeseries) * train_ratio)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    return train, test

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X).float().reshape([-1, lookback, 1]), torch.tensor(y).float().reshape([-1, lookback, 1])

def train_model(
    model, optimizer, criterion, 
    X_train, y_train, 
    X_test, y_test, n_epochs, batch_size, verbose=False, report_every=20,
):
    """
        Given an LSTM model architecture, trains it and saves the weights.
    """
    # get data loader
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    # store training and test error 
    train_error, test_error = [], []
    for epoch in range(n_epochs):
        # turn on training mode
        model.train()
        optimizer.zero_grad()
        for X_batch, y_batch in loader:
            # make a prediction on the batch
            y_pred = model(X_batch)
            # evaluate training loss on the batch 
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # reporting 
        if epoch % report_every != 0:
            continue 
        model.eval()
        with torch.no_grad():
            # compute training error on all data
            y_pred = model(X_train)
            train_rmse = np.sqrt(criterion(y_pred, y_train))
            # compute test error on all data
            y_pred = model(X_test)
            test_rmse = np.sqrt(criterion(y_pred, y_test))
            if verbose:
                print(">    Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
        # save error
        train_error.append(train_rmse.item())
        test_error.append(test_rmse.item())
    return train_error, test_error
        
##########
class BasicLSTM(nn.Module):
    """ 
        Base version of LSTM
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.name = "BasicLSTM"
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class StackedLSTM(nn.Module):
    """ 
        Three LSTMs stacked together.
    """
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.name = "StackedLSTM"
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.linear(x)
        return x

class EnsembleLSTM(nn.Module):
    """ 
        Several basic LSTMs are trained separately on the same dataset 
        (potentially with bootstrap resampling), and the prediction 
        of all models is combined using a final linear layer. 
    """
    def __init__(self, num_models):
        super().__init__()
        self.num_models = num_models
        self.lstms = nn.ModuleList([BasicLSTM() for _ in range(num_models)])
        self.linear = nn.Linear(num_models, 1)
        self.name = "EnsembleLSTM"
    def forward(self, x):
        outputs = []
        for lstm in self.lstms:
            outputs.append(lstm(x))
        outputs = torch.stack(outputs, dim=1)
        # permute dimensions to expose to the linear layer application
        outputs = torch.permute(outputs, (0, 2, 3, 1))
        outputs = self.linear(outputs)
        outputs = outputs.squeeze(3)
        return outputs

class AttentionLSTM(nn.Module):
    """ 
        A basic LSTM model with attention layer.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size + input_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_size, 1)
        self.name = "AttentionLSTM"

    def forward(self, x):
        # Pass the input through the LSTM
        lstm_out, _ = self.lstm(x)

        # Concatenate the LSTM output and the input
        concat = torch.cat((lstm_out, x), dim=2)

        # Apply the attention layer
        attention_weights = torch.softmax(self.attention(concat), dim=1)

        # Multiply the attention weights with the LSTM output
        weighted_output = torch.bmm(attention_weights.unsqueeze(1).squeeze(3), lstm_out)

        # Apply the dropout layer
        dropped = self.dropout(weighted_output)

        # Apply the final linear layer
        output = self.linear(dropped)

        return output

