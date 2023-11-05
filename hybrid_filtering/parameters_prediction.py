from typing import Any
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

class Params_prediction:
    def __init__(self, num_param, lr=0.0001, pred_type="item") -> None:
        """
        This class will aim to predict some uparameters based on the features provided
        """
        self.learning_rate = lr
        self.create_model(num_param)

        self.type = pred_type

        self.model_path = f"hybrid_filtering/models/parameters_predictor/{self.type}"
    def create_model(self, num_param):
        self.model = tf.keras.Sequential([
            Dense(256, "relu"),
            Dense(256, "relu"),
            Dense(num_param, "linear"),
        ])


        self.model.compile(optimizer=Adam(self.learning_rate), loss=MeanSquaredError(), metrics=['accuracy'])


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
    
    def save_model(self):
        self.model.save(self.model_path)
        print(f"paramets predictor {self.type} saved")

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"paramets predictor {self.type} loaded")
    

