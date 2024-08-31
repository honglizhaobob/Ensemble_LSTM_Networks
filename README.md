# Ensemble_LSTM_Networks
---


### Running the script

1. First run ```python3 load_data.py``` to download the necessary dataset from Yahoo Finance. Please navigate to ```/experiments/``` to see sample EDA of the downloaded dataset.

2. Run ```python3 main.py``` to train all LSTM models considered:
	* Base LSTM
	* Stacked (3 models) LSTM
	* Ensemble (bagged 5 models) LSTM
	* LSTM with Attention Layer
The trained model weights and loss information will be saved in ```/saved_models/```. To load and test the models, run ```python3 predict.py```.
	
An example base LSTM output is shown here for 18 stocks:
![alt text](/asset/lstm2.png)

The model architectures are visualized below:
![alt text](/asset/lstm_output.png)

Finally, the general runtime for training 500 epochs are shown below:
<img src="/asset/wall_clock_time.png" alt="alt text" width="400">

Derivations, statistical testing, and more information can be found from the [write-up](https://honglizhaobob.github.io/projects/lstm/).
