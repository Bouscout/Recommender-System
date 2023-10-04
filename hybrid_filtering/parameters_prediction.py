from typing import Any
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

class Params_prediction:
    def __init__(self, num_param, lr=0.0001) -> None:
        """
        This class will aim to predict some uparameters based on the features provided
        """
        self.learning_rate = lr
        self.create_model(num_param)
        

    def create_model(self, num_param):
        self.model = tf.keras.Sequential([
            Dense(256, "relu"),
            Dense(256, "relu"),
            Dense(num_param, "linear"),
        ])


        self.model.compile(optimizer=Adam(self.learning_rate), loss=MeanSquaredError())

    def prediction(self, features):
        """
        Make a prediction of a parameters for the NCF system\n

        Input must be passed with the same normalization and scaling applied to it during training
        """
        params = self.model(features)
        return params
    
    def train(self, feature, params, epochs:int=1, verbose=False):
        self.model.fit(feature, params, epochs=epochs, verbose=verbose)

    def __call__(self, feature_vec) -> Any:
        return self.prediction(feature_vec)
