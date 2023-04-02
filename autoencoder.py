# an LSTM autoencoder

import tensorflow as tf
import numpy as np

class LSTM_Autoencoder(tf.keras.layers.Layer):
    def __init__(self, timesteps, features, encoding_dim, name='lstm_autoencoder'):
        super(LSTM_Autoencoder, self).__init__(name=name)
        self.timesteps = timesteps
        self.features = features
        self.encoding_dim = encoding_dim
        # encoding
        self.lstm1 = tf.keras.layers.LSTM(self.encoding_dim, activation='relu', return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(self.encoding_dim, activation='relu', return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(self.encoding_dim // 2, activation='relu', return_sequences=True)
        # decoding
        self.lstm4 = tf.keras.layers.LSTM(self.encoding_dim // 2, activation='relu', return_sequences=True)
        self.lstm5 = tf.keras.layers.LSTM(self.encoding_dim, activation='relu', return_sequences=True)
        self.lstm6 = tf.keras.layers.LSTM(self.features, activation='sigmoid', return_sequences=True)
        
    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        x = self.lstm5(x)
        x = self.lstm6(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    