# how does it work ?

# from a series of rating given by a number Nu user to a series of movies of number Nm
# we want to find a set of paramters w, b for each user with same dimensions as feature of any movie of shape (x)

# the equation to find the compability to a movie will be : rating = (w dot x) + b

# from these information derive the set of parameters that match the given movie set for each user
# from these parameters we can derive the set of feature X that defines a movie
# from these parameters we can predict the set of rating that a new user might give to a new movie

import tensorflow as tf
import numpy as np
np.random.seed(5)
# feature dimensions
X_dimensions = 5
X_n = 20 # num of users
X_m = 10 # num of movies

movie_features = np.random.randn(X_m, X_dimensions)
movie_ratings = np.random.randint(1, 6, size=(X_m, X_n))

rated = np.random.randint(0, 2, size=(X_m, X_n))

user_weights = np.random.randn(X_n, X_dimensions)
user_bias = np.random.randn(1, X_n)

# amount of movie not rated by user

# get the database of movies and feature
label = tf.convert_to_tensor(movie_ratings)
movie_parameters = tf.Variable(movie_features)

# get the database of user
w = tf.Variable(user_weights)
b = tf.Variable(user_bias)

def find_Movie_Norm_Mean(rating, rated):
    # find the mean with only the rated movies
    movie_rated = rating * rated

    mean_ratings = np.sum(movie_rated, axis=1)
    num_of_rated = np.sum(rated, axis=1)

    mean_ratings = mean_ratings / num_of_rated

    normalized_rating = movie_rated - mean_ratings[:, None]

    return normalized_rating, mean_ratings

# calculate the loss
def loss_func(x, w, b, y, r, _lambda_=1.5):
    estim = tf.linalg.matmul(x, tf.transpose(w)) 
    test = np.matmul(movie_parameters, user_weights.T)
    estim_w_bias = estim + b
    loss_unfiltered = (estim_w_bias - y)**2

    loss_regular = loss_unfiltered * r

    weight_part = (_lambda_ / 2) * tf.reduce_sum(w**2)
    input_part = (_lambda_ / 2) * tf.reduce_sum(x**2)

    regularization = weight_part + input_part

    loss = 0.5 * tf.reduce_sum(loss_regular) + regularization

    return loss 

# prepare the model 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 4000

label, mean = find_Movie_Norm_Mean(movie_ratings, rated)

# fit the movie, user
for iteration in range(epochs) :
    with tf.GradientTape() as tape :
        loss = loss_func(movie_parameters, w, b, label, rated)


    gradient = tape.gradient(loss, [w, b, movie_parameters])
    optimizer.apply_gradients(zip(gradient, [w, b, movie_parameters]))

    if iteration % 20 == 0 : 
        print("for epoch ", iteration, " the loss : ", loss)

# check accuracy

predicted_rating = (tf.linalg.matmul(movie_parameters, tf.transpose(w))) + b
predicted_rating += mean[:, None]

difference = (movie_ratings * rated) - (predicted_rating * rated)

print(difference)
print("the inaccuracy is : ", np.sum(difference.numpy()**2) / len(difference))