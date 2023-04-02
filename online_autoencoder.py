from autoencoder import LSTM_Autoencoder

import tensorflow as tf
import numpy as np
from queue import Queue


class OnlineLSTMAutoencoder(tf.keras.layers.Layer):
    def __init__(self, timesteps, features, encoding_dim, tolerance=0.1, name='online_lstm_autoencoder'):
        super(OnlineLSTMAutoencoder, self).__init__(name=name)
        self.timesteps = timesteps
        self.features = features
        self.encoding_dim = encoding_dim
        self.buffer = Queue(maxsize=timesteps)
        self.n = 0
        self.tolerance = tolerance
        self.b = self.timesteps ** 0.5
        self.mean = np.zeros((features,))
        self.var = np.zeros((features,))  # variation
        self.s = np.zeros((features,))  # auxiliary variable for the variance
        self.lstm_encoder = LSTM_Autoencoder(self.timesteps, self.features, self.encoding_dim)

    def call(self, x):
        self.buffer.put_nowait(x)

        if self.buffer.full():
            mean = np.mean(self.buffer, axis=0)  # the old mean
            # get the first element of the buffer
            o = self.buffer.get_nowait()
            self.n += 1

            # update the mean and the variance
            mean_new = mean + (o - mean) / self.n

            self.s = self.s + np.multiply((o - mean), (o - mean_new))
            var_new = np.sqrt(self.s / self.n)

            std_err = self.var / self.b
            z = (x - mean) / std_err

            if np.abs(z) > self.tolerance:
                self.n -= 1
                print("Anomaly detected!")
                print(x)
                # reject the new data point
                return self.lstm_encoder(tf.zeros_like(x))
            else:
                # update the mean and the variance
                self.mean = mean_new
                self.var = var_new

                # encode the data
                return self.lstm_encoder(x)
        
        return self.lstm_encoder(tf.zeros_like(x))
    
    def compute_output_shape(self, input_shape):
        return input_shape


class ReconstructionLoss(tf.keras.losses.MeanSquaredError):
    def __init__(self, name='reconstruction_loss'):
        super(ReconstructionLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        if tf.reduce_sum(y_pred) == 0:
            return tf.zeros((1,))
        else:
            return super(ReconstructionLoss, self).call(y_true, y_pred)
