import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import pickle

class Neural_Collaborative_filtering():
    def __init__(self, num_user, num_item, x_dim, learning_rate=0.0001, _lambda_=1.5, output_dim=1) -> None:
        """
        Implementation of a collaborative system using a neural network to predict
        the affinity between a user and an item vector.\n

        algorithm will train some user and item parameters represent the latent vector of the relationship between all users and all movies\n

        those latent variables are accessible through a getter function and could be used for other purpose

        required : \n 
        the num of user and item \n
        a dimension number x_dim for the number of features into the user and item vectors\n
        a set of rating or label representing the user interaction or affinity with the item\n
        a set or rating_mask representing in binary form which item has been rated by which users\n

        hyperparameters :
        learning_rate, epochs, lambda\n
        output_dim : represent the num of prediction on the last layer
        usually is 1 but could be a larger number to handle operations with other models

        """
        self.X_u = num_user
        self.X_i = num_item
        
        self.x_dim = x_dim

        self._lambda_ = _lambda_

        # preparing the learnable variables
        # self._user_params = tf.Variable(tf.random.normal(shape=(num_user, x_dim), dtype=tf.float32))
        # self._item_params = tf.Variable(tf.random.normal(shape=(num_item, x_dim), dtype=tf.float32))

        # we need to keep ip as numpy arrays so that we can index through different user item index combinaisons
        # without the need to create copy of matrix all the time
        self._user_params = np.random.randn(num_user, x_dim).astype(np.float32)
        self._item_params = np.random.randn(num_item, x_dim).astype(np.float32)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(output_dim, "linear"),
        ])

        self.model.compile(optimizer=Adam(learning_rate), loss="MSE", metrics=['accuracy'])

        self.optimizer = Adam(learning_rate=learning_rate)
        
        self.mean = 0

        self.model_path = "hybrid_filtering/models/ncf/ncf_model"

        self.user_variable_path = "hybrid_filtering/models/ncf/user_params.pickle"
        self.item_variable_path = "hybrid_filtering/models/ncf/item_params.pickle"

    def loss_func(self, w, x, y_true, r):
        # concat the input and item vector before passing them to the network
        # expanding and repeating the arguments for concatenation
        num_user, num_item = w.shape[0], x.shape[0]
        w = tf.repeat(tf.expand_dims(w, axis=1), num_item, axis=1)
        x = tf.repeat(tf.expand_dims(x, axis=0), num_user, axis=0)

        input_vector_concat = tf.concat([w, x], axis=-1)
        
        # flattening the input vector
        input_vector_concat = tf.reshape(input_vector_concat, (num_user * num_item, self.x_dim*2))

        # flattening the label
        y_true = np.reshape(y_true, (-1, 1))
        r = np.reshape(r, (-1, 1))

        y_predict = self.model(input_vector_concat)

        loss_regular = (y_predict - y_true)**2
        loss_regular *= r  # filtering non rated item

        # regularization
        user_param = (self._lambda_ / 2) * (tf.reduce_sum(w**2)) 
        item_param = (self._lambda_ / 2) * (tf.reduce_sum(x**2)) 

        regularization = user_param + item_param

        loss = 0.5 * tf.reduce_sum(loss_regular) + regularization

        return loss
    
    def normalize_label(self, all_ratings, rating_mask, num_user, num_movie):
        """
        Normalize the ratings by substracting the mean

        return the normalize values and set the mean value to substract from future predictions
        """
        # in order to get the mean per movie and not per user
        # flattening
        all_ratings = all_ratings.reshape(num_user, num_movie)
        rating_mask = rating_mask.reshape(num_user, num_movie)

        # transposing
        all_ratings = all_ratings.T
        rating_mask = rating_mask.T

        epsilon = 1e-6
        ratings = all_ratings * rating_mask

        num_rating_per_row = np.sum(rating_mask, axis=1) # total num of rating per row

        # mean calculation
        mean_rating = np.sum(ratings, axis=1)
        mean_rating = mean_rating / (num_rating_per_row + epsilon) 

        normalized_rating = ratings - mean_rating[:, None]

        self.mean = mean_rating[:, None] # setting the mean

        return normalized_rating.astype(np.float32)

    def train(self,  rating, rating_mask, indexes=None, epochs=200, verbose=False, normalize=True) :
        """
        Perform training over the given batch for a number of epochs
        """
        # getting the params
        if indexes :
            w, x = self.get_params_from_idx(indexes)
            w = tf.Variable(w)
            x = tf.Variable(x)
        else :
            w = tf.Variable(self._user_params)
            x = tf.Variable(self._item_params)

        # normalizing the labels
        if normalize :
            num_user, num_movie = w.shape[0], x.shape[0]
            label = self.normalize_label(rating, rating_mask, num_user, num_movie)
        else :
            label = rating

        for iteration in range(epochs) :
            with tf.GradientTape() as tape :
                loss = self.loss_func(w, x, label, rating_mask)

            gradient = tape.gradient(loss, [w, x, *self.model.trainable_variables])

            self.optimizer.apply_gradients(zip(gradient, [w, x, *self.model.trainable_variables]))

            if iteration % 20 == 0 and verbose :
                print("for iter : ", iteration, " loss is : ", loss)

        # setting the new params
        if indexes :
            self.set_params_at_idx(w.numpy(), x.numpy(), indexes)
        else :
            self._user_params = w.numpy()
            self._item_params = x.numpy()
        
        return loss
    
    def get_params(self):
        """
        return the user and item parameters
        """
        return (self._user_params, self._item_params)
    
    def get_params_from_idx(self, indexes:tuple) :
        """
        Return the User and item param at the given indexes from the stored parameters
        """
        user_idx, item_idx = indexes
        return (self._user_params[user_idx], self._item_params[item_idx])
        # return (tf.gather(self._user_params, user_idx), tf.gather(self._item_params, item_idx))

    def set_params_at_idx(self, u_param, i_param, indexes:tuple):
        """
        Set the param at the given indexes to the new values given if possible
        """
        user_dim, item_dim = u_param.shape[-1], i_param.shape[-1]
        if user_dim != item_dim :
            raise ValueError("The dimensions at axis(1) do not match")
        
        else :
            user_idx, item_idx = indexes
            self._user_params[user_idx] = u_param
            self._item_params[item_idx] = i_param

            return True

    def set_params(self, user_param, item_param):
        """
        Set the user and item parameters to given values and then set the informatives variables with respect to the new params\n

        return error if user and param dimensions are not appropriate
        """
        num_user, user_dim = user_param.shape 
        num_item, item_dim = item_param.shape

        if user_dim != item_dim :
            raise ValueError("The dimensions at axis(1) do not match")
        
        else :
            # changing information variable
            self.X_u = num_user
            self.X_i = num_item
            self.x_dim = user_dim

            self._user_params = user_param
            self._item_params = item_param

            return True

    def expand_dim_pair(self, u_param, i_param):
        num_user, num_item = u_param.shape[0], i_param.shape[0]

        u_param = tf.repeat(tf.expand_dims(u_param, axis=1), num_item, axis=1)
        i_param = tf.repeat(tf.expand_dims(i_param, axis=0), num_user, axis=0)
        
        return u_param, i_param

    def predict(self, user_param=None, item_param=None, indexes=None, expand=False, normalize=True) :
        """
        Return the prediction of the model after unapplying the normalization process
        from the previous training batch\n

        use object.raw_predict() to get the raw prediction from the model
        """
        if indexes :
            user_param, item_param = self.get_params_from_idx(indexes)
            num_user = user_param.shape[0]
            movie_idx = indexes[1]
            

        if expand :
            user_param, item_param = self.expand_dim_pair(user_param, item_param)

        input_vector = np.concatenate([user_param, item_param], axis=-1)

        input_vector = tf.reshape(input_vector, (-1, input_vector.shape[-1]))

        y_pred = self.model(input_vector)

        if normalize :
            if indexes :
                mean = self.mean[movie_idx].reshape(-1, 1)
                mean = np.tile(mean, (num_user, 1))
                y_pred += mean # in order to cancel the normalization

            else :
                y_pred += self.mean

        return y_pred.numpy()
    
           
        
    def raw_predict(self, user_param=None, item_param=None, indexes=None ,flatten=True, expand=False) :
        """
        return the raw prediction of the model\n 

        if the rank of both params > 2, it will concatenate then appropriately
        before computing and returning the prediction matrix
        """
        if indexes :
            user_param, item_param = self.get_params_from_idx(indexes)

        user_rank, item_rank = len(user_param.shape), len(item_param.shape)
        if user_rank != item_rank :
            raise ValueError("the two set of parameters don't have the same rank, they cannot be concated appropriatly")

        if expand :
            user_param, item_param = self.expand_dim_pair(user_param, item_param)

        input_vector = tf.concat([user_param, item_param], axis=-1)

        input_vector = tf.reshape(input_vector, (-1, input_vector.shape[-1]))

        prediction = self.model(input_vector)
        if not flatten :
            # unflatten
            prediction = tf.reshape(prediction, (user_param.shape[0], user_param.shape[1]))

        return prediction

    def save_model(self):
        self.model.save(self.model_path)

        with open(self.user_variable_path, "wb") as fichier :
            pickle.dump(self._user_params, fichier)

        with open(self.item_variable_path, "wb") as fichier :
            pickle.dump(self._item_params, fichier)

        print("NCF saved")

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

        with open(self.user_variable_path, "rb") as fichier :
            self._user_params = pickle.load(fichier)

        with open(self.item_variable_path, "rb") as fichier :
            self._item_params = pickle.load(fichier)

        self.X_u = self._user_params.shape[0]
        self.X_i = self._item_params.shape[0]
        self.x_dim = self._item_params.shape[1]

        print("ncf loaded")

