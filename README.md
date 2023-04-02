# Hack Kosice 2023 - UPJS Challenge

## Unsupervised anomaly detection with RNN autoencoders based on LSTM cells, with some statistical data filtering

### The Task:

The task is to detect anomalies in a given dataset(`./data/dc_file_modified2.csv`). The dataset is represented of a time series of a Windows computer logs. There has been an attack on this computer, which has exposed some precious data. The task is to detect the anomalies in the dataset, which are the signs of the attack.

### The Solution:

The solution is based on the unsupervised anomaly detection with RNN autoencoders based on LSTM cells. The model learns to predict the logs, i.e. the activity on the computer. The model is trained on the healthy logs only, because we don't want it to predict the anomalies as well. During training, some of the data does not contribute to the loss, because it deviates too much from the last `timesteps` logs.


### Techincal details:

The model is implemented in tensorflow 2.