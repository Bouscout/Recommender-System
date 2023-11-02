from hybrid_filtering.NCF import Neural_Collaborative_filtering
from hybrid_filtering.content_based_filtering import Content_Based_filtering
from hybrid_filtering.hybrid_recommendation_system import Hybrid_recommendation_system
from hybrid_filtering.parameters_prediction import Params_prediction
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
np.random.seed(5)
tf.random.set_seed(5)

# with this test, we evaluate the performance on a set new movies
# we will have a model prediction the params of the movie in the collaborative filtering based on the pattern observed from 
# the training data

class Recommender_System:
    def __init__(self, filter:Hybrid_recommendation_system, epochs) -> None:
        self.filter = filter
        self.epochs = epochs

    def train(self,xu, xi, y, r, index):
        loss = self.filter.train(xu, xi, index, y, r, epochs=self.epochs, verbose=False, expand=True)
        return loss

def main(u_start, u_end, movie_start, movie_end, filter:Recommender_System):
    global dataset, user_feat, movie_feat, train_movie_rating

    rating = train_movie_rating[u_start:u_end, movie_start:movie_end].reshape(-1, 1)
    mask = np.where(rating.reshape(-1, 1) >= 0, 1, 0)


    # selecting the training range of user and movies
    xu = user_feat[u_start:u_end]
    xm = movie_feat[movie_start:movie_end]  

    user_index = np.arange(u_start, u_end)
    movie_index = np.arange(movie_start, movie_end)

    loss = filter.train(xu, xm, rating, mask, (user_index, movie_index))

    return loss

def evaluate(u_start, u_end, movie_start, movie_end, recommend:Hybrid_recommendation_system):
   
    global test_movie_rating, user_feat, test_movie_feat, movie_feat, mask, test_mask
    xu = user_feat[u_start:u_end]
    xm = test_movie_feat
  

    user_index = np.arange(u_start, u_end)
    user_param, _ = recommend.ncf.get_params_from_idx((user_index, None))

    # we predict the movie parameters
    movie_param =  movie_param_predictor(recommend.predict_latent_vec(test_movie_feat))   

    prediction = recommend.prediction(xu, xm, u_params=user_param, i_params=movie_param, expand=True)
   
    prediction_reverse = prediction.numpy()

    prediction_reverse = prediction_reverse 

    print("===========================================")
    print("the prediction is :")
    print((prediction_reverse  *test_mask).reshape(len(xu), len(xm)))
    print("===========================================")
    
    difference = (prediction_reverse) - test_movie_rating * test_mask
   

    print("===========================================")

    print("the label is : ")
    print(((test_movie_rating) * test_mask).reshape(len(xu), len(xm)))

    print("===========================================")
    boolean_mask = test_mask > 0
    loss = difference[boolean_mask]
    print("the overall loss is : ", np.sum(loss**2))
    print()



dataset = pd.read_csv("my_anime_data_cleaned/rating_matrix_v1.csv").iloc[:, 1:] # we increment to leave the movieId

dataset = dataset.to_numpy()
num_movie, num_user = dataset.shape


test = 1500 # test size

test_movie_rating = dataset[num_movie-test:].T
test_mask = np.where(test_movie_rating.reshape(-1, 1) >= 0, 1, 0)

train_movie_rating = dataset[:num_movie - test].T

# dataset = dataset[:].T # we transpose for shape (num_user, num_movie)

# preparing and cleaning the anime features
movie_feat = pd.read_csv("my_anime_data_cleaned/anime_data.csv", sep=",")
# movie_feat.drop(columns=["start_date", "end_date"], inplace=True)


# preparing the user feat
user_feat = pd.read_csv("my_anime_data_cleaned/user_features.csv", sep=",")

def retrieve_anime_index(anime_ids:str) -> list:
    ids = anime_ids.split("|")
    ids = ids[:-1] # last one is empty string
    ids = list(map(int, ids))

    indexes = movie_feat.loc[movie_feat["anime_id"].isin(ids), "anime_index"]
    return list(indexes)


def decaying_average(anime_indexes:list) -> np.ndarray :
    decay_rate = 0.95
    num_series = len(anime_indexes)

    decay_factor = np.array([decay_rate**x for x in range(num_series)]).reshape(-1, 1)

    series_features = movie_feat[anime_indexes]

    series_features *= decay_factor

    return np.mean(series_features, axis=0)    


user_anime_indexes = [0 for _ in range(user_feat.shape[0])]
for i in range(len(user_feat)) :
    watched = user_feat.loc[i, "watched"]
    user_anime_indexes[i] = retrieve_anime_index(watched)


# preparing the samples
movie_feat = movie_feat.iloc[:, 2:].to_numpy()

test_movie_feat = movie_feat[num_movie-test:num_movie]
train_movie_feat = movie_feat[:-test]

user_feat = user_feat.to_numpy()

# preparing the scalers
# xu_scaler = StandardScaler()
# xm_scaler = StandardScaler()
xm_scaler = MinMaxScaler()
# labelScaler = MinMaxScaler()

# scaling the inputs vectors
# xu_scaler.fit(user_feat)
# user_feat = xu_scaler.transform(user_feat)

xm_scaler.fit(movie_feat)
train_movie_feat = xm_scaler.transform(train_movie_feat)
test_movie_feat = xm_scaler.transform(test_movie_feat)

mask = np.where(train_movie_rating.reshape(-1, 1) >= 0, 1, 0)
# labelScaler.fit(dataset.reshape(-1, 1))

dataset = train_movie_rating.reshape(-1, 1) # the ratings

test_movie_rating = test_movie_rating.reshape(-1, 1)

user_feat = np.zeros((num_user, movie_feat.shape[-1]))
for i in range(len(user_anime_indexes)) :
    user_feat[i] = decaying_average(user_anime_indexes[i])

# print(user_feat[0])

u_dim = user_feat.shape[-1]
m_dim = movie_feat.shape[-1]

movie_feat = train_movie_feat

lr = 0.0001

step = 30

filter_system = Hybrid_recommendation_system(num_user, num_movie, u_dim, m_dim, lr=lr, l_d=128, binary=True)
recommender = Recommender_System(filter_system, 1)

random_u_start = num_user - 100
random_m_start = num_movie - 100

# the model to predict the movie parameters inside the collaborative filtering system
movie_param_predictor = Params_prediction(filter_system.ncf.x_dim)

# evaluate(0, num_user, num_movie-test, num_movie, filter_system)

# print("===============================")
# input("press to start training...")

print("starting to train")
epochs = 20
num_movie_train = num_movie - test

user_batch_size = 50
movie_batch_size = 100
try : 
    for _ in range(epochs):
        for u_batch in range(user_batch_size, num_user, user_batch_size) :
            for m_batch in range(movie_batch_size, num_movie_train, movie_batch_size) :
                # main(0, num_user, 0, num_movie-test, recommender)

                u_start = u_batch - user_batch_size
                m_start = m_batch - movie_batch_size

                loss = main(u_start, u_batch, m_start, m_batch, recommender)

                print(f"epochs {_} u_batch {u_batch} m_batch {m_batch} loss : {loss}", end="\r")

        print(f"epochs {_} u_batch {u_batch} m_batch {m_batch} loss : {loss}")
except :
        print(f"epochs {_} u_batch {u_batch} m_batch {m_batch} loss : {loss}")


movie_idx = np.arange(0, num_movie-test)



user_param, movie_param = filter_system.ncf.get_params_from_idx((np.arange(0, 2), movie_idx))
movie_vec = filter_system.predict_latent_vec(movie_feat)

movie_param_predictor.train(movie_vec, movie_param, epochs=100, verbose=False)


evaluate(0, num_user, num_movie-test, num_movie, filter_system)

    