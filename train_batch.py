import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from hybrid_filtering.content_based_filtering import Content_Based_filtering
from hybrid_filtering.hybrid_recommendation_system import Hybrid_recommendation_system
import tensorflow as tf

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf

def train(u_start, u_end, m_start, m_end, recommender:Hybrid_recommendation_system):
    # import data
    hybrid = True

    user_feature = pd.read_csv("cleaned_data/user_info.csv").iloc[u_start:u_end]
    movie_feature = pd.read_csv("cleaned_data/movie_info.csv").iloc[m_start:m_end]
    rating_matrix = pd.read_csv("cleaned_data/movie_rating.csv").iloc[m_start:m_end, u_start:u_end + 1]

    # find info variable
    num_user, user_feat_dim = user_feature.shape
    num_movie, movie_feat_dim = movie_feature.shape

    user_feat_dim -= 1
    movie_feat_dim -= 1 

    # print(f"num user : {num_user}\t user feature dimension : {user_feat_dim}")
    # print(f"num movie : {num_movie}\t movie feature dimension : {user_feat_dim}")

    rating_matrix = rating_matrix.iloc[:, 1:].to_numpy()
    rating_matrix = rating_matrix.T.reshape(-1, 1)
    rating_mask = np.where(rating_matrix > -1, 1, 0)

    user_feature = user_feature.iloc[:, 1:].to_numpy()
    movie_feature = movie_feature.iloc[:, 1:].to_numpy()

    
    print(f"num user : {num_user}\t user feature dimension : {user_feat_dim}")
    print(f"num movie : {num_movie}\t movie feature dimension : {user_feat_dim}")


    # normalize data
    userScaler = StandardScaler()
    movieScaler = StandardScaler()
    rating_scaler = MinMaxScaler()

    userScaler.fit(user_feature)
    user_feature = userScaler.transform(user_feature)

    movieScaler.fit(movie_feature)
    movie_feature = movieScaler.transform(movie_feature)

    rating_scaler.fit(rating_matrix)
    rating_matrix = rating_scaler.transform(rating_matrix)

    if hybrid :

        # split data
        train_size = 1.0

        # userTrain, userTest = train_test_split(user_feature, train_size=train_size, shuffle=True, random_state=1)
        # movieTrain, movieTest = train_test_split(movie_feature, train_size=train_size, shuffle=True, random_state=1)
        # ratingTrain, ratingTest = train_test_split(np.reshape(rating_matrix.T, (-1, 1)), train_size=train_size, shuffle=True, random_state=1)
        # ratingMaskTrain, ratingMaskTest = train_test_split(np.reshape(rating_mask.T, (-1, 1)), train_size=train_size, shuffle=True, random_state=1)

        num_user_batch, user_dim = user_feature.shape
        num_movie_batch, movie_dim = movie_feature.shape
        # num_user_batch, user_dim = userTrain.shape
        # num_movie_batch, movie_dim = movieTrain.shape
        # train over data
        
        # recommender.train_recommender(userTrain, movieTrain, ratingTrain, ratingMaskTrain, verbose=True, expand=True)
        param_index = (np.arange(u_start, u_end), np.arange(m_start, m_end))
        recommender.train_recommender(user_feature, movie_feature, param_index, rating_matrix, rating_mask, epochs=100,verbose=True, expand=True)

def evaluate(u_start, u_end, m_start, m_end, recommender:Hybrid_recommendation_system):

    
    # import data
    user_feature = pd.read_csv("cleaned_data/user_info.csv").iloc[u_start:u_end]
    movie_feature = pd.read_csv("cleaned_data/movie_info.csv").iloc[m_start:m_end]
    rating_matrix = pd.read_csv("cleaned_data/movie_rating.csv").iloc[m_start:m_end, u_start:u_end+1]

    # find info variable
    num_user, user_feat_dim = user_feature.shape
    num_movie, movie_feat_dim = movie_feature.shape

    user_feat_dim -= 1
    movie_feat_dim -= 1 

    rating_matrix = rating_matrix.iloc[:, 1:].to_numpy()

    rating_matrix = rating_matrix.T.reshape(-1, 1)
    rating_mask = np.where(rating_matrix > -1, 1, 0)



    user_feature = user_feature.iloc[:, 1:].to_numpy()


    movie_feature = movie_feature.iloc[:, 1:].to_numpy()


    # normalize data
    userScaler = StandardScaler()
    movieScaler = StandardScaler()
    rating_scaler = MinMaxScaler()

    userScaler.fit(user_feature)
    user_feature = userScaler.transform(user_feature)

    movieScaler.fit(movie_feature)
    movie_feature = movieScaler.transform(movie_feature)

    rating_scaler.fit(rating_matrix)
    # rating_matrix = rating_scaler.transform(rating_matrix)

    print("for the training randomly selected variable : ")
    print("============================================================")

    param_index = (np.arange(u_start, u_end), np.arange(m_start, m_end))

    prediction = recommender.prediction(user_feature, movie_feature, indexes=param_index, expand=True)

    # prediction = np.reshape(prediction.numpy(), (num_user, num_movie))
    prediction_reverse = rating_scaler.inverse_transform(prediction)

    difference = prediction_reverse - rating_matrix

    prediction_matrix = prediction_reverse.reshape(num_user, num_movie) * rating_mask.reshape(num_user, num_movie)

    print(prediction_reverse.reshape(num_user, num_movie).astype(np.int8) * rating_mask.reshape(num_user, num_movie))

    print("============================================================")
    print(difference.reshape(num_user, num_movie) * rating_mask.reshape(num_user, num_movie))
    print("============================================================")
    print("the right answer is  : ")
    print(rating_matrix.reshape(num_user, num_movie) * rating_mask.reshape(num_user, num_movie))
    # recommender.train_recommender(user_feature, movie_feature, param_index,rating_matrix.T, rating_mask.T, epochs=1,verbose=True, expand=True)
    print("============================================================")
    

# test accuracy
if __name__ == "__main__" :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("gpu set up")
        except RuntimeError as e:
            print(e)
    
    
    sys_details = tf.sysconfig.get_build_info()
    cuda_version = sys_details["cuda_version"]
    print(cuda_version)


    total_num_user = pd.read_csv("cleaned_data/user_info.csv").shape[0] 
    total_num_movie = pd.read_csv("cleaned_data/movie_info.csv").shape[0] 

    total_num_movie-= 4000
    
    user_dimension = 22
    movie_dimension = 22

    class content_based:
        def __init__(self) -> None:
            self.filter = Content_Based_filtering(22, 22, 32)

        def train(self, xu, xi, y, r):
            self.filter.train(xu, xi, y, r)

        def evaluate(self, xu, xi, y, r):
            pass

    step = total_num_user - 1
    # step = 50

    recommender = Hybrid_recommendation_system(total_num_user, total_num_movie, user_dimension, movie_dimension, l_d=64)
    recommender.weights = (0.5, 0.5)

    # every step amount of user for all movies
    epochs = 3
    for _ in range(epochs) :
        movie_start = 0
        user_start = 0
        for user_end in range(step, total_num_user, step) :

            for movie_end in range(step, total_num_movie, step) :
            # for movie_end in range(20) :
                train(user_start, user_end, movie_start, movie_end, recommender=recommender)
                # train(0, 15, 0, 15, recommender=recommender)

                movie_start = movie_end

                print("done : ", movie_end)

            user_start = user_end

            print("============================")

    random_u_start = np.random.randint(0, total_num_user - 10)
    random_m_start = np.random.randint(0, 500)

    # random_u_start = 0
    # random_m_start = 0

    evaluate(random_u_start, random_u_start+10, random_m_start, random_m_start+10, recommender)
    