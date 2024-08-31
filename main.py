import os
from models import *
from plotting_helper import *

# Select a stock to test
TICKER = "NVDA"
# testing period (k-th day)
k = 1512 # 2021-04-30

# helper script to load data saved locally
def load_data(tk):
    dir_name = "./data"
    filename = [s for s in os.listdir(dir_name) if tk in s]
    if not filename:
        raise ValueError("Data for {} does not exist. ".format(tk))
    # assume only one file
    filename = filename[0]
    # split for reporting
    tmp = filename.split("_")
    ticker, start_date, end_date = tmp[0], tmp[1], tmp[-1].split(".")[0]
    print("Loaded data for {} from {} to {}".format(
        ticker, start_date, end_date
    ))
    data = pd.read_csv("./data/"+filename)
    assert not data.isnull().values.any()
    return data

if __name__ == "__main__":
    # checks data exists and loads data
    data = load_data(TICKER)
    
    # use adjusted close
    date = data["Date"][:k]
    price = data["Adj Close"][:k]
    # report trained date range
    print("> Experiment date range = {} TO {}".format(date.min(), date.max()))
    price_data = (date, price)

    # split training and testing data for PyTorch
    train, test = split_timeseries(price)
    plot_train_test_split(TICKER, train, test)

    # create dataset
    lookback = 10
    X_train, y_train = create_dataset(train.values, lookback=lookback)
    X_test, y_test = create_dataset(test.values, lookback=lookback)
    # start training (create four models)
    all_models = [
        BasicLSTM(), StackedLSTM(), EnsembleLSTM(num_models=5), AttentionLSTM(1, 50, 3)
    ]
    for model in all_models:
        print(">    Training {} for Ticker = {}".format(model.name, TICKER))
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        train_error, test_error = train_model(
            model, optimizer, loss_fn, 
            X_train, y_train, X_test, y_test, 
            n_epochs=500, batch_size=2**3, verbose=True, report_every=20,
        )
        # save loss and model 
        with open(f"./saved_models/{model.name}_{TICKER}_train.pkl", 'wb') as f:
            pickle.dump(train_error, f)
        with open(f"./saved_models/{model.name}_{TICKER}_test.pkl", 'wb') as f:
            pickle.dump(test_error, f)
        print("Loss information saved")
        torch.save(model.state_dict(), f"./saved_models/{model.name}_{TICKER}.pt")
        print("Model weights saved. ")


