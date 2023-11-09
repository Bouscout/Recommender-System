from hybrid_filtering_pytorch.ncf_torch import Neural_Collaborative_filtering
from hybrid_filtering_pytorch.content_filtering_torch import Content_Based_filtering
# from hybrid_filtering.hybrid_recommendation_system import Hybrid_recommendation_system
from hybrid_filtering_pytorch.hybrid_recommender_torch import Hybrid_recommendation_system
from hybrid_filtering.parameters_prediction import Params_prediction
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import pandas as pd
import numpy as np
import os

np.random.seed(5)
# tf.random.set_seed(5)



# with this test, we evaluate the performance on a set new movies
# we will have a model prediction the params of the movie in the collaborative filtering based on the pattern observed from 
# the training data

class Recommender_System:
    def __init__(self, filter:Content_Based_filtering, epochs) -> None:
        self.filter = filter
        self.epochs = epochs

    def train(self,xu, xi, rating, mask, index):
        loss = self.filter.train(xu, xi, index, rating, mask, epochs=self.epochs, verbose=False, expand=True)
        # loss = self.filter.train(y, r, index, epochs=self.epochs, verbose=False)
        # loss = self.filter.train(xu, xi, rating, mask, self.epochs, verbose=False)
        return loss

def main(u_start, u_end, movie_start, movie_end, filter:Recommender_System):
    global dataset, user_feat, train_movie_feat, dataset, labelScaler

    rating = dataset.T[u_start:u_end, movie_start:movie_end].reshape(-1, 1)
    mask = np.where(rating >= 0, 1, 0)

    rating = labelScaler.transform(rating)

    # selecting the training range of user and movies
    xu = user_feat[u_start:u_end]
    xm = train_movie_feat[movie_start:movie_end]  

    user_index = np.arange(u_start, u_end)
    movie_index = np.arange(movie_start, movie_end)

    loss = filter.train(xu, xm, rating, mask, (user_index, movie_index))

    return loss

def evaluate(u_start, u_end, movie_start, movie_end, recommend:Content_Based_filtering):
   
    global user_feat, test_movie_feat, dataset, labelScaler

    test_movie_rating = dataset.T[u_start:u_end, movie_start:movie_end].reshape(-1, 1)
    test_mask = np.where(test_movie_rating >= 0, 1, 0)

    xu = user_feat[u_start:u_end]
    xm = test_movie_feat
  
    user_index = np.arange(u_start, u_end)
    movie_index = np.arange(movie_start, movie_end)
    user_param, movie_param = recommend.ncf.get_params_from_idx((user_index, movie_index))

    # we predict the movie parameters
    # movie_param =  movie_param_predictor(recommend.predict_latent_vec(test_movie_feat))   

    # converting to tensor
    user_param = torch.tensor(user_param, dtype=torch.float32, device=recommend.device)
    movie_param = torch.tensor(movie_param, dtype=torch.float32, device=recommend.device)

    xu = torch.tensor(xu, dtype=torch.float32, device=recommend.device)
    xm = torch.tensor(xm, dtype=torch.float32, device=recommend.device)

    

    prediction = recommend.prediction(xu, xm, u_params=user_param, i_params=movie_param, expand=True)
    # prediction = recommend.raw_predict(indexes=(user_index, movie_index), expand=True)
    # prediction = recommend.predict(xu, xm, expand=True)
   
    # prediction_reverse = prediction.detach().cpu().numpy()
    prediction_reverse = labelScaler.inverse_transform(prediction.detach().cpu().numpy())

    boolean_mask = test_mask > 0

    print("===========================================")
    print("the prediction is :")
    print((prediction_reverse[boolean_mask][-10:]))
    print("===========================================")
    
    difference = (prediction_reverse - test_movie_rating) * test_mask
   

    print("===========================================")

    print("the label is : ")
    print((test_movie_rating[boolean_mask][-10:]))

    print("===========================================")
    loss = difference[boolean_mask]
    print("the overall loss is : ", np.sum(loss**2))
    print()

def experiment_with_indexes(user_index:int, anime_index:int) -> tuple:
    # will return the predicted rating and the actual rating given a user and anime information
    global movie_feat, dataset, user_columns, recommender, labelScaler
    user_features = pd.read_csv("my_anime_data_cleaned/user_data_v2.csv").loc[int(user_index), "watched"]
    watched_anime = retrieve_anime_index(user_features)

    user_features = decaying_average(watched_anime).reshape(1, -1)
    anime_features = movie_feat[anime_index].reshape(1, -1)

    column_index = user_columns.index(user_index)

    predicted = recommender.filter.prediction(user_features, anime_features, indexes=([column_index], [anime_index]))
    reverse_predicted = labelScaler.inverse_transform(predicted.numpy())

    true_rating = dataset[anime_index, column_index]

    return (reverse_predicted, true_rating)





import pickle
def save_checkpoint_info():
    with open("checkpoint.pickle", "wb") as f :
        save_point = {
            "epoch" : epoch ,
            "loss" : overall_loss,
        }
        pickle.dump(save_point, f)


last_epoch = 0
previous_loss = 1000
def load_save_point(new=False):
    global recommender, last_epoch, overall_loss
    recommender.filter.load_model()

    with open("checkpoint.pickle", "rb") as f :
        save_point = pickle.load(f)
    
    if new :
        last_epoch = 0
    else :
        last_epoch = save_point["epoch"]
        overall_loss = save_point["loss"]

    movie_param_predictor.load_model()

    print("precedent session loaded")



dataset = pd.read_csv("my_anime_data_cleaned/rating_matrix_v2.csv").iloc[:, 1:] # we increment to leave the movieId

user_columns = list(dataset.columns)

dataset = dataset.to_numpy()
num_movie, num_user = dataset.shape

test = 1000 # test size

# dataset = dataset[:].T # we transpose for shape (num_user, num_movie)

# preparing and cleaning the anime features
movie_feat = pd.read_csv("my_anime_data_cleaned/anime_data.csv", sep=",")
# movie_feat.drop(columns=["start_date", "end_date"], inplace=True)


# preparing the user feat
user_feat = pd.read_csv("my_anime_data_cleaned/user_features.csv", sep=",")

def retrieve_anime_index(anime_ids:str, index=False) -> list:
    ids = anime_ids.split("|")
    ids = ids[:-1] # last one is empty string
    ids = list(map(int, ids))

    if isinstance(movie_feat, pd.DataFrame) :
        indexes = movie_feat.loc[movie_feat["anime_id"].isin(ids), "anime_index"]
    else :
        df = pd.read_csv("my_anime_data_cleaned/anime_data.csv")
        indexes = df.loc[df["anime_id"].isin(ids), "anime_index"]

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
xm_scaler = MinMaxScaler()
labelScaler = MinMaxScaler()

# anime input vectors
xm_scaler.fit(train_movie_feat)
train_movie_feat = xm_scaler.transform(train_movie_feat)
test_movie_feat = xm_scaler.transform(test_movie_feat)

# user input vectors
user_feat = np.zeros((num_user, movie_feat.shape[-1]))
for i in range(len(user_anime_indexes)) :
    user_feat[i] = decaying_average(user_anime_indexes[i])

# label input vectors
labelScaler.fit(dataset.T.reshape(-1, 1)[dataset.T.reshape(-1, 1) >= 0].reshape(-1, 1))

# print(user_feat[0])

u_dim = user_feat.shape[-1]
m_dim = movie_feat.shape[-1]

# movie_feat = train_movie_feat

lr = 0.00001
overall_loss = 10000

step = 5
random_u_start = num_user - 100
random_m_start = num_movie - 100

filter_system = Hybrid_recommendation_system(num_user, num_movie, u_dim, m_dim, lr=lr, l_d=128)
# filter_system = Neural_Collaborative_filtering(num_user, num_movie, x_dim=128, learning_rate=lr)
# filter_system = Content_Based_filtering(83, 83, 128)
recommender = Recommender_System(filter_system, 1)

recommender.filter.load_model("1")

evaluate(300, 400, 3000, 4000, filter_system)

# for e in range(50):
#     total_loss = 0
#     for i in range(5) :
#         deb = 3000 + (500 * i)
#         end = deb + 450

#         l = main(0, 572, deb, end, recommender)
#         total_loss += l
#         print(f"epoch {e+1}, limit {end}, loss : {l}", end='\r')

#     print(f"epoch {e+1}, loss : {total_loss}")

# the model to predict the movie parameters inside the collaborative filtering system
# movie_param_predictor = Params_prediction(filter_system.ncf.x_dim)

# load_save_point(new=True)

# while True :
#     u_index = str(input("enter user index : "))
#     a_index = int(input("enter anime index : "))
#     answer = experiment_with_indexes(u_index, a_index)
#     print(f"for u:{u_index} and a:{a_index} the predictions are : ", answer, end="")
# evaluate(200, 400, num_movie-test, num_movie, filter_system)
# evaluate(300, 400, 6000, 7000, filter_system)

# recommender.filter.save_model("1")

# l = main(200, 311, 3800, 4000, recommender)
# print("loss is : ", l)

def check_up():
    print("===============================")

    check_up = filter_system.hybrid_model.l1.weight[0][:10]
    print("check up : ", check_up)

    check_up_ncf = filter_system.ncf._user_params[0, 0:10]
    print("check up user : ", check_up_ncf)

    input("press to start training...")


epochs = 20
num_movie_train = num_movie - test

user_batch_size = 572
movie_batch_size = 450

# saving the scalers
import pickle
with open("scalers/item_scaler.pickle", "wb") as f :
    pickle.dump(xm_scaler, f)

with open("scalers/label_scaler.pickle", "wb") as f :
    pickle.dump(labelScaler, f)

    
# load_save_point(new=True)

print("starting to train")
try : 
    for epoch in range(epochs):
        total_loss = 0
        if epoch < last_epoch :
            continue
        for u_batch in range(user_batch_size, num_user, user_batch_size) :
            for m_batch in range(movie_batch_size, num_movie_train, movie_batch_size) :
                # main(0, num_user, 0, num_movie-test, recommender)

                u_start = u_batch - user_batch_size
                m_start = m_batch - movie_batch_size

                loss = main(u_start, u_batch, m_start, m_batch, recommender)
                
                total_loss += loss

                print(f"epochs {epoch} u_batch {u_batch} m_batch {m_batch} loss : {loss}", end="\r")
            
       


        print(f"epochs {epoch} loss : {total_loss}")

        if total_loss < overall_loss :
            recommender.filter.save_model(model_set="1")
            overall_loss = total_loss
            save_checkpoint_info()


except Exception as e:
    print(e)
    print(f"epochs {epoch} u_batch {u_batch} m_batch {m_batch} loss : {loss}")

  

print("---------------------------------")
print("training the predictor")
movie_idx = np.arange(0, num_movie-test)
user_idx = np.arange(0, num_user - 50)

movie_param_predictor = Params_prediction(filter_system.ncf.x_dim)
# movie_param_predictor.load_model()

user_params, movie_param = filter_system.ncf.get_params_from_idx((user_idx, movie_idx))
movie_vec = filter_system.predict_latent_vec(train_movie_feat, isUser=False)

movie_param_predictor.train(movie_vec, movie_param, epochs=200, verbose=True)

movie_param_predictor.save_model()

user_param_predictor = Params_prediction(recommender.filter.ncf.x_dim, pred_type="user")
# user_param_predictor.load_model()

user_vec = recommender.filter.predict_latent_vec(user_feat[user_idx])

user_param_predictor.train(user_vec, user_params, epochs=200, verbose=True)
user_param_predictor.save_model()


evaluate(200, 400, num_movie-test, num_movie, filter_system)
    