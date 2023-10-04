# we will implement a collaborative filtering but this time using a neural network
import tensorflow as tf
import numpy as np
np.random.seed(5)
tf.random.set_seed(5)
# generate the data
# feature dimensions
X_dimensions = 5
X_n = 20 # num of users
X_m = 10 # num of movies

movie_features = np.random.randn(X_m, X_dimensions).astype(np.float32)
movie_ratings = np.random.randint(1, 6, size=(X_m, X_n))

rated = np.random.randint(0, 2, size=(X_m, X_n))

user_weights = np.random.randn(X_n, X_dimensions).astype(np.float32)

# initialize parameters
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, "relu"),
    tf.keras.layers.Dense(128, "relu"),
    tf.keras.layers.Dense(1, "linear"),
])


user_parameters = tf.Variable(user_weights)
movie_params = tf.Variable(movie_features)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def find_ItemNorm_Mean(item, rated):
    ratings = item * rated
    num_rating_per_row = np.sum(rated, axis=1)
    mean_rating = np.sum(ratings, axis=1)

    mean_rating = mean_rating / num_rating_per_row

    normalized_rating = ratings - mean_rating[:, None]

    return normalized_rating.astype(np.float32), mean_rating.astype(np.float32)

def loss_func(x, w, y_true_matrix, r,_lambda_=1.5) :
    # each user with respect to every movies
    # input_vector = np.zeros((X_n, X_m, X_dimensions*2))

    w_expanded = np.repeat(w[:, np.newaxis, :], X_m, axis=1)
    x_expanded = np.repeat(x[None, :, :], X_n, axis=0)
    input_vector = np.concatenate([w_expanded, x_expanded], axis=-1) 

    input_flattened = tf.reshape(input_vector, (X_n * X_m, X_dimensions*2))

    y_predict = model(input_flattened)

    y_true = tf.reshape(y_true_matrix, (X_n * X_m, 1))

    loss_regular = (y_predict - y_true)**2
    loss_regular *= r.reshape(-1, 1)

    weight_part = tf.reduce_sum(w**2) * (_lambda_ / 2)
    input_part = tf.reduce_sum(x**2) * (_lambda_ / 2)
    regularization = weight_part + input_part

    loss = 0.5 * tf.reduce_sum(loss_regular) + regularization

    return loss

# perform training
epochs = 4000
label, mean = find_ItemNorm_Mean(movie_ratings, rated)

previous_loss = 1000

for iteration in range(epochs):
    with tf.GradientTape() as tape :
        loss = loss_func(movie_params, user_parameters, label, rated)

    gradient = tape.gradient(loss, [movie_params, user_parameters, *model.trainable_variables])
    optimizer.apply_gradients(zip(gradient, [movie_params, user_parameters, *model.trainable_variables]))

    if iteration % 20 == 0 : 
        print("for epoch ", iteration, " the loss : ", loss)

    if iteration > 200 :
        if previous_loss < loss :
            break

        previous_loss = loss

# check accuracy

input_vector = np.zeros((X_n, X_m, X_dimensions*2))
for i in range(X_n):
        for j in range(X_m):
            concat = np.concatenate([user_parameters[i], movie_params[j]])
            input_vector[i, j] = concat

predicted_rating = model(input_vector)

predicted_rating = np.reshape(predicted_rating.numpy(), (X_m, X_n))

predicted_rating += mean[:, None]

difference = ((movie_ratings * rated) - (predicted_rating * rated))

print(difference)
print("the inaccuracy is : ", np.sum(difference ** 2) / len(difference))