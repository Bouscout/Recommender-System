# In content based filtering, we will use highly detailed set of feature for the item and the user
# the goal is to be able to predict the affinity (rating) of a user to a specific item

# we will need a model capable a user vector representing the user affinity
# we will need the same type of model but this time for the movie

# the operation between those two latent vectors would output the affinity of the user to the given item

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
np.random.seed(5)

# generate data
Xu_dimension = 10
Xi_dimension = 22
latent_vector_size = 10

num_item = 30
num_user = 12

movie_feature = np.random.randn(num_item, Xi_dimension)

user_feature = np.random.randn(num_user, Xu_dimension)

ratings = np.random.randint(0, 6, (num_item, num_user))
rated = np.random.randint(0, 2, (num_item, num_user))

scalerItem = StandardScaler()
scalerUser = StandardScaler()
scalerLabel = MinMaxScaler()

scalerUser.fit(user_feature)
scaled_Xu = scalerUser.transform(user_feature)

scalerItem.fit(movie_feature)
scaled_Xi = scalerItem.transform(movie_feature)

scalerLabel.fit(ratings)
scaled_label = scalerLabel.transform(ratings)


# prepare the model
userModel = tf.keras.Sequential([
    tf.keras.layers.Dense(128, "relu"),
    tf.keras.layers.Dense(128, "relu"),
    tf.keras.layers.Dense(latent_vector_size, "linear"),
])
input_user = tf.keras.layers.Input(shape=(Xu_dimension,))
Vu = userModel(input_user)
Vu = tf.linalg.l2_normalize(Vu, axis=-1)

itemModel = tf.keras.Sequential([
    tf.keras.layers.Dense(128, "relu"),
    tf.keras.layers.Dense(128, "relu"),
    tf.keras.layers.Dense(latent_vector_size, "linear"),
])
input_item = tf.keras.layers.Input(shape=(Xi_dimension,))
Vi = itemModel(input_item)
Vi = tf.linalg.l2_normalize(Vi, axis=-1)

output = tf.keras.layers.Dot(axes=1)([Vu, Vi])
# output = tf.linalg.matmul(Vu, Vi)

model = tf.keras.Model([input_user, input_item], output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_criterion = tf.keras.losses.MeanSquaredError()
# train model
def loss_func(Xu, Xi, r, y) :
    num_user, num_movie = len(Xu), len(Xi)
    # concat each user for all movies
    expand_Xu = tf.repeat(tf.expand_dims(Xu, axis=1), num_movie, axis=1)
    expand_Xi = tf.repeat(tf.expand_dims(Xi, axis=0), num_user, axis=0)

    Xu = tf.reshape(expand_Xu, (num_user * num_movie, Xu.shape[-1]))
    Xi = tf.reshape(expand_Xi, (num_movie * num_user, Xi.shape[-1]))


    y_pred = model([Xu, Xi])

    y = np.reshape(y, (-1, 1))
    r = np.reshape(r, (-1, 1))

    loss = (y_pred - y)**2
    # Apply the mask to the squared error to filter out non-rated items
    filtered_squared_error = tf.boolean_mask(loss, r)

    return tf.reduce_mean(filtered_squared_error)

epochs = 500
transpose_rated = rated.T
transpose_rating = scaled_label.T



previous_loss = 10000
for iteration in range(epochs) :

    with tf.GradientTape() as Tape :
        loss = loss_func(scaled_Xu, scaled_Xi, rated, scaled_label)

    gradient = Tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    if iteration % 20 == 0 : 
        print("for epoch ", iteration, " the loss : ", loss)

    if iteration > 200 :
        if previous_loss < loss :
            break

        previous_loss = loss

# check accuracy
# let's check prediction for a random user
predicted_rating = np.zeros_like(scaled_label).T
for idx in range(len(predicted_rating)) :
    predicted_rating[idx] = model([np.tile(scaled_Xu[idx], (num_item, 1)), scaled_Xi]).numpy()[:, 0]

predicted_rating = predicted_rating

predicted_rating = scalerLabel.inverse_transform(predicted_rating.T)



difference = ((predicted_rating * rated) - (ratings * rated))

print(difference)
print("the inaccuracy is : ", np.sum(difference ** 2) / len(difference))
