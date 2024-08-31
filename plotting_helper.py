# Helper functions to generate and save pre-defined visualizations (used in main.py)
import matplotlib.pyplot as plt

def plot_train_test_split(ticker, train, test):
    plt.figure(figsize=(15, 6))
    # Plot the training data in blue
    plt.plot(train, color="blue", label="Training Data")
    # Plot the test data in red
    plt.plot(test, color="red", label="Test Data")
    # Create shaded areas for the windows in the training data
    train_size = len(train)
    for i in range(5):
        start_index = int(i * train_size / 5)
        end_index = int((i + 1) * train_size / 5)
        if i == 0:
            plt.axvspan(start_index, end_index, alpha=0.2, color="gray", label="Training windows")
        else:
            plt.axvspan(start_index, end_index, alpha=0.2, color="gray", label="")

    # Add a legend and title
    plt.legend(fontsize=16)
    plt.title("{} Stock Price with Training and Test Data".format(ticker), fontsize=16)
    plt.xlabel("Days", fontsize=16)
    plt.ylabel("Price", fontsize=16)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.savefig("./fig/{}_training_data.png".format(ticker), dpi=200)