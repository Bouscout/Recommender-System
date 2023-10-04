import tensorflow as tf
from keras.optimizers import Adam
import numpy as np

class Content_Based_filtering():
    def __init__(self, Xu, Xi, V_dim, lr=0.0001, special_dimension=None) -> None:
        """
        Implementation of Content Based filtering which uses two neural network to represent two latent vector Vu and Vi
        of dimentions dim representing the latent variable of the different feature of the user and item input\n

        a matrix multiplication between those two latent vector represent the affinity or rating of the user toward that item
        required : \n
        user input vector dimension : Xu\n
        item input vector dimension : Xi\n
        latent vector dimension : Vm\n

        hyperparameters : learning_rate
        """
        self.create_models(Xu, Xi, V_dim, lr, special_dimension)

        self.model_path = "hybrid_filtering/models/content_filter/content_model"

    def create_models(self, Xu, Xi, Vm, lr, special_dimension):    
        # user model
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(Vm, "linear"),
        ])
        input_user = tf.keras.layers.Input(shape=(Xu, ))
        Vu = self.user_model(input_user)
        Vu = tf.linalg.l2_normalize(Vu, axis=1)
      
        # user model
        self.item_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(Vm, "linear"),
        ])
        input_item = tf.keras.layers.Input(shape=(Xi,))
        Vi = self.item_model(input_item)
        Vi = tf.linalg.l2_normalize(Vi, axis=1)

        # combining the models
        output_num = special_dimension if special_dimension else 1
        # means we want the to produce a latent vector for future interaction
        latent_vector_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(output_num, "linear"),
        ])

        special_input = tf.concat([Vu, Vi], axis=-1)
        output = latent_vector_model(special_input)

      
        # output = tf.keras.layers.Dot(axes=1)([Vu, Vi])

        self.model = tf.keras.Model([input_user, input_item], output)
        self.optimizer = Adam(learning_rate=lr)

        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())


    def loss_func(self, Xu, Xi, r, y) :
        num_user, num_movie = len(Xu), len(Xi)
        # concat each user for all movies
        expand_Xu = tf.repeat(tf.expand_dims(Xu, axis=1), num_movie, axis=1)
        expand_Xi = tf.repeat(tf.expand_dims(Xi, axis=0), num_user, axis=0)

        # flattening the extra dimensions
        Xu = tf.reshape(expand_Xu, (-1, Xu.shape[-1]))
        Xi = tf.reshape(expand_Xi, (-1, Xi.shape[-1]))

        # both model prediction
        y_pred = self.model([Xu, Xi])

        y = np.reshape(y, (-1, 1))
        r = np.reshape(r, (-1, 1))

        loss = (y_pred - y)**2
        # Apply the mask to the squared error to filter out non-rated items
        filtered_squared_error = tf.boolean_mask(loss, r)
        # filtered_squared_error = loss * r

        return tf.reduce_sum(filtered_squared_error)
    
    def expand_dim_pair(self, u_param, i_param):
        num_user, num_item = len(u_param), len(i_param)

        u_param = tf.repeat(tf.expand_dims(u_param, axis=1), num_item, axis=1)
        i_param = tf.repeat(tf.expand_dims(i_param, axis=0), num_user, axis=0)
        
        return u_param, i_param

    def train(self, user_input, item_input, rating, rating_mask, epochs=200, verbose=False) :
        """
        Perform training over the given batch for a number of epochs
        """
        # we use the transpose version because I compute the loss from the perspective of each user
        # with respect to all items
        if rating.shape[-1] != 1 :
            transpose_rating = rating.T
            tranpose_r = rating_mask.T
        else :
            transpose_rating = rating
            tranpose_r = rating_mask

        previous_loss = 10000

        for iteration in range(epochs) :
            
            with tf.GradientTape() as tape :
                loss = self.loss_func(user_input, item_input, tranpose_r, transpose_rating)

            # gradient step
            gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

            if iteration % 20 == 0 and verbose :
                print("for epoch ", iteration, " the loss : ", loss)

            if iteration > 200 :
                if previous_loss < loss :
                    print("early stop at iteration : ", iteration)
                    break

                previous_loss = loss

        

    def predict(self, user_input, item_input, flatten=True, expand=False):
        """
        Make a prediction based on given user and item feature\n

        if they have a rank > 2, meaning we want to perform the prediction with respect to all items for every user
        we will return a matrix representing the rating for each user movie pair
        """
        user_rank, item_rank = len(user_input.shape), len(item_input.shape)
        if user_rank != item_rank :
            raise ValueError("the two set of features don't have the same rank")
        
        if expand :
            user_input, item_input = self.expand_dim_pair(user_input, item_input)
            user_rank = 3
        
        if user_rank > 2 :
            num_user, num_item = user_input.shape[0], user_input.shape[1] 
            # we need to flatten then before making prediction
            rating = self.model([
                tf.reshape(user_input, (-1, user_input.shape[-1])),
                tf.reshape(item_input, (-1, item_input.shape[-1])),
            ])

            if not flatten :
                return tf.reshape(rating, (num_user, num_item))
            else :
                return rating
        
        rating = self.model([user_input, item_input])
        return rating
    
    def save_model(self):
        self.model.save(self.model_path)
        print("content model saved")

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print("content model loaded")