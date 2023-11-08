# we will be using a combination of NCF and content based filtering to produce a prediction
# of the user affinity with a certain item

# we will used a weighted sum method to get our final prediction in order to prevent the
# cold start problem with our limited user data

import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from hybrid_filtering.NCF import Neural_Collaborative_filtering
from hybrid_filtering.content_based_filtering import Content_Based_filtering
np.random.seed(5)
tf.random.set_seed(5)

# prepare the data
user_info_path = "cleaned_data/user_info.csv"
movie_info_path = "cleaned_data/movie_info.csv"

class Hybrid_recommendation_system():
    def __init__(self, num_user:int, num_item:int, u_feat_dim:int, i_feat_dim:int, *, lr:float=0.001, l_d=20, binary=False) -> None:
        """
        Hybrid recommendation system based on the use of a Neural Collaborative filtering system and a content based recommender system\n

        Address combined prediction by the use of linear weighted sum allowing to tackle the problem of cold start with small values\n
        N.B : 
        Data normalization and batch separation should be handled outside this class \n

        simple prediction should be limited to inputs of shape(batch_size, feature_size) and should always have the same number of item \n

        training can handle different size inputs and will perform loss calculation for each user-item pair provided\n

        loss is calculated from the perspective of the user with respect to all item so rating should be passed in shape (num_user, num_item)\n
        same thing for the rating mask shape (num_user, num_item), generally just require a transpose function
        """
        self.num_user = num_user
        self.num_item = num_item

        Xu_dim = u_feat_dim # user feature dimension 
        Xi_dim = i_feat_dim # item feature dimension

        # vector dimension outputed by both recommender system
        latent_vector_dimension = l_d

        # initialize hyperparameters and model parameters
        self.epochs = 200
        self.learning_rate = lr
        self.weights = (0.5, 0.5)
        self.optimizer = Adam(learning_rate=lr)
        self.binary = binary
        self.create_model(binary)

        # prepare the NCF algorithm
        collab_feature_dimension = 128
        self.ncf = Neural_Collaborative_filtering(self.num_user, self.num_item, collab_feature_dimension, output_dim=latent_vector_dimension)

        self.user_param, self.item_param = self.ncf.get_params()

        # prepare the content based algorithm
        model_dim = 64
        self.content_filtering = Content_Based_filtering(Xu_dim, Xi_dim, model_dim, special_dimension=latent_vector_dimension)

        self.model_path = "hybrid_filtering/models/hybrid/hybrid_model"

    def create_model(self, binary):
        self.hybrid_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dense(1, "sigmoid" if binary else "linear"),
        ])

        self.binary = binary

    def prediction(self, u_features, i_features, u_params=None, i_params=None, indexes=None, weights:tuple=None, expand=False):
        """
        Make a prediction of the affinity of a user to a certain item using movie and item features and parameters\n

        if value where normalized during training, make sure to apply a reverse process to the prediction in order to get appropriate value\n

        return >>> prediction shape(batch_size, 1)
        """
        weights = weights if weights else self.weights

        ncf_prediction = self.ncf.raw_predict(u_params, i_params, indexes=indexes ,expand=expand)
        c_recommender_pred = self.content_filtering.predict(u_features, i_features, expand=expand)

        if np.any(np.isnan(ncf_prediction)) or np.any(np.isinf(ncf_prediction)) :
            print("nan in ncf")

        if np.any(np.isnan(c_recommender_pred)) or np.any(np.isinf(c_recommender_pred)) :
            print("nan in c_recommend")

        # normalize prediction
        ncf_prediction = tf.linalg.l2_normalize(ncf_prediction, axis=1)
        c_recommender_pred = tf.linalg.l2_normalize(c_recommender_pred, axis=1)

        ncf_weight, c_recom_weight = weights

        hybrid_input = tf.concat([ncf_prediction*ncf_weight, c_recommender_pred*c_recom_weight], axis=1)

        output = self.hybrid_model(hybrid_input)
        
        return output

    # train the models
    def train(self, u_features, i_features, param_idx ,ratings, rating_mask, expand=False, epochs=0, verbose=False) :
        """
        Compute loss with respect to combine recommendation of both recommender system
        and then backpropagate the gradient with respect to all learables variables in the overall system
        """
        previous_loss = 0
        self.num_item = len(i_features)
        self.num_user = len(u_features)

        # getting the user and item parameters
        u_param, i_param = self.ncf.get_params_from_idx(param_idx)
        u_param = tf.Variable(u_param)
        i_param = tf.Variable(i_param)

        # function to get all the params
        def all_trainables_params() :
            params = [u_param, i_param, *self.ncf.model.trainable_variables, 
                      *self.content_filtering.model.trainable_variables, *self.hybrid_model.trainable_variables
                      ]
            return params
        
        def loss_func(u_feat, i_feat, ratings, rating_mask):
            epsilon = 1e-6

            ncf_weight, content_r_weight = self.weights 

            output = self.prediction(u_feat, i_feat, u_params=u_param, i_params=i_param, weights=(ncf_weight, content_r_weight), expand=expand)

            if self.binary :
                filtered_output = tf.boolean_mask(output, rating_mask)
                filtered_rating = tf.boolean_mask(ratings, rating_mask)

                # test_output = filtered_output.numpy()
                # test_rating = filtered_rating.numpy()

                # if np.any(np.isnan(test_output)) or np.any(np.isinf(test_output)):
                #     print("there is a nan")

                filtered_output = tf.clip_by_value(filtered_output, epsilon, 1-epsilon) # to avoid nan value

                loss= tf.keras.losses.BinaryCrossentropy(from_logits=False)(filtered_rating, filtered_output)

                return loss
            
            else :
                ratings = np.reshape(ratings, (-1, 1))
                rating_mask = np.reshape(rating_mask, (-1, 1))

                squared_error = (output - ratings)**2

                # Apply the mask to the squared error to filter out non-rated items
                filtered_squared_error = tf.boolean_mask(squared_error, rating_mask)

                # return tf.reduce_mean(filtered_squared_error)
                return tf.reduce_sum(filtered_squared_error)

        # initializing new optimizer cause we might be using new params for the ncf
        optimizer = Adam(learning_rate=self.learning_rate)            

        # training process
        training_epochs = epochs if epochs else self.epochs
        for iteration in range(training_epochs) :
            with tf.GradientTape() as tape :
                loss = loss_func(u_features, i_features, ratings, rating_mask)

            # gradient step
            params = all_trainables_params()
            gradient = tape.gradient(loss, params)
            optimizer.apply_gradients(zip(gradient, params))

            if verbose and iteration % 20 == 0 :
                print("for epoch ", iteration, " the loss : ", loss.numpy())

                if iteration > 200 :
                    if previous_loss < loss :
                        print("early stop at iteration : ", iteration)
                        break

                    previous_loss = loss

        # setting the new user params at the indexes
        self.ncf.set_params_at_idx(u_param.numpy(), i_param.numpy(), param_idx)
        
        return loss.numpy()


    def predict_latent_vec(self, features, isUser=True):
        """Make a prediction a parameters using the content filter parameters vector and an output model"""
        if isUser :
            latent_features = self.content_filtering.user_model(features)
            return latent_features
        
        else :
            latent_features = self.content_filtering.item_model(features)
            return latent_features
        
    def save_model(self):
        self.ncf.save_model()
        self.content_filtering.save_model()

        self.hybrid_model.save(self.model_path)
        print("all model saved")

    def load_model(self):
        self.hybrid_model = tf.keras.models.load_model(self.model_path)

        self.ncf.load_model()
        self.content_filtering.load_model()

        # loading info variables
        self.num_user = self.ncf._user_params.shape[0]
        self.num_item = self.ncf._item_params.shape[0]

        print("all models loaded")


# test accuracy