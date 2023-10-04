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
        # self.filter.train(y, r, index, epochs=self.epochs, verbose=True, normalize=False)
        # self.filter.train(xu, xi, y, r, self.epochs, verbose=True)
        self.filter.train(xu, xi, index, y, r, epochs=self.epochs, verbose=True, expand=True)



def main(u_start, u_end, movie_start, movie_end, filter:Recommender_System):
    global dataset, user_feat, movie_feat
   

    rating = dataset

    xu = user_feat[u_start:u_end]
    xm = movie_feat[movie_start:movie_end]

  

    user_index = np.arange(u_start, u_end)
    movie_index = np.arange(movie_start, movie_end)

    # filter.train(rating, mask, index=(user_index, movie_index))
    filter.train(xu, xm, rating, mask, (user_index, movie_index))
    # filter.train(xu, xm, rating, mask)

def evaluate(u_start, u_end, movie_start, movie_end, recommend:Hybrid_recommendation_system):
   
    global test_movie_rating, user_feat, test_movie_feat, movie_feat, mask, test_mask
    xu = user_feat
    xm = test_movie_feat
  

    user_index = np.arange(u_start, u_end)
    movie_index = np.arange(movie_start, movie_end)

    user_param, _ = recommend.ncf.get_params_from_idx((user_index, None))
    movie_param =  movie_param_predictor(recommend.predict_latent_vec(test_movie_feat))   

    # prediction = recommend.predict(indexes=(user_index, movie_index), expand=True, normalize=False)
    # prediction = recommend.prediction(xu, xm, indexes=(user_index, movie_index), expand=True)
    prediction = recommend.prediction(xu, xm, u_params=user_param, i_params=movie_param, expand=True)
    # prediction = recommend.predict(xu, xm, expand=True)
    # prediction_reverse = prediction
    prediction_reverse = prediction.numpy()
    # prediction_reverse = labelScaler.inverse_transform(prediction_reverse)

    prediction_reverse = prediction_reverse 
    # prediction_reverse = prediction_reverse.reshape(len(xu), len(xm)) * test_mask.reshape(len(xu), len(xm))

    print("===========================================")
    print("the prediction is :")
    # print(prediction_reverse.reshape(len(xu), len(xm)))
    print((labelScaler.inverse_transform(prediction_reverse)  *test_mask).reshape(len(xu), len(xm)))
    print("===========================================")
    
    # difference = (prediction_reverse - test_movie_rating) * test_mask
    difference = (labelScaler.inverse_transform(prediction_reverse) - labelScaler.inverse_transform(test_movie_rating)) * test_mask
    # difference = (prediction_reverse - test_movie_rating.reshape(len(xu), len(xm))) * test_mask.reshape(len(xu), len(xm))
    # print("the difference is :")
    # print(difference.reshape(len(xu), len(xm)))

    print("===========================================")

    print("the answer is : ")
    # print(test_movie_rating.reshape(len(xu), len(xm)) * test_mask.reshape(len(xu), len(xm)))
    print((labelScaler.inverse_transform(test_movie_rating) * test_mask).reshape(len(xu), len(xm)))

    print("===========================================")
    boolean_mask = test_mask > 0
    loss = difference[boolean_mask]
    print("the overall loss is : ", np.mean(loss**2))
    print()



dataset = pd.read_csv("cleaned_data/trimmed_ratings.csv").iloc[:, 1:] # we increment to leave the movieId
dataset = dataset.to_numpy()
num_movie, num_user = dataset.shape

test = 50 # test size

test_movie_rating = dataset[num_movie-test:].T
test_mask = np.where(test_movie_rating.reshape(-1, 1) >= 0, 1, 0)

train_movie = dataset[:num_movie - test].T

dataset = dataset[:].T # we transpose for shape (num_user, num_movie)

# we increment both by 1 to avoid taking in the ids
movie_feat = pd.read_csv("cleaned_data/movie_info_100.csv").iloc[:num_movie - test, 1:] 
user_feat = pd.read_csv("cleaned_data/user_info_200.csv").iloc[:num_user, 1:]

test_movie_feat = pd.read_csv("cleaned_data/movie_info_100.csv").iloc[num_movie-test:num_movie, 1:]

movie_feat = movie_feat.to_numpy()
user_feat = user_feat.to_numpy()

xu_scaler = StandardScaler()
xm_scaler = StandardScaler()
labelScaler = MinMaxScaler()

xu_scaler.fit(user_feat)
user_feat = xu_scaler.transform(user_feat)

xm_scaler.fit(movie_feat)
movie_feat = xm_scaler.transform(movie_feat)
test_movie_feat = xm_scaler.transform(test_movie_feat)

mask = np.where(train_movie.reshape(-1, 1) >= 0, 1, 0)
labelScaler.fit(dataset.reshape(-1, 1))
# dataset = labelScaler.transform(dataset.reshape(-1, 1))
dataset = labelScaler.transform(train_movie.reshape(-1, 1))

# test_mask = mask[-(num_user * 30):]
# test_movie_rating = dataset[-(num_user * 30):]
test_movie_rating = labelScaler.transform(test_movie_rating.reshape(-1, 1))
# test_mask = mask[num_user*10:]
# dataset = dataset[:-(num_user * 30)]
# mask = mask[:-(num_user * 30)]

print(user_feat[0])

u_dim = user_feat.shape[-1]
m_dim = movie_feat.shape[-1]

lr = 0.00001

step = 30

# filter_system = Content_Based_filtering(u_dim, m_dim, 128, lr=lr)
# filter_system = Neural_Collaborative_filtering(num_user, num_movie, 64, learning_rate=0.000001)
filter_system = Hybrid_recommendation_system(num_user, num_movie, u_dim, m_dim, lr=lr, l_d=128)
recommender = Recommender_System(filter_system, 1)

random_u_start = num_user - 30
random_m_start = num_movie - 30

movie_param_predictor = Params_prediction(filter_system.ncf.x_dim)

evaluate(0, num_user, num_movie-test, num_movie, filter_system)

print("===============================")
print("starting to train")
for _ in range(100):
    main(0, num_user, 0, num_movie-test, recommender)


movie_idx = np.arange(0, num_movie-test)



user_param, movie_param = filter_system.ncf.get_params_from_idx((np.arange(0, 2), movie_idx))
movie_vec = filter_system.predict_latent_vec(movie_feat)

movie_param_predictor.train(movie_vec, movie_param, epochs=100, verbose=False)


evaluate(0, num_user, num_movie-test, num_movie, filter_system)

    