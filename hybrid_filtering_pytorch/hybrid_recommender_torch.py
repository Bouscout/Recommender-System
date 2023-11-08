# we will be using a combination of NCF and content based filtering to produce a prediction
# of the user affinity with a certain item

# we will used a weighted sum method to get our final prediction in order to prevent the
# cold start problem with our limited user data

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hybrid_filtering_pytorch.ncf_torch import Neural_Collaborative_filtering
from hybrid_filtering_pytorch.content_filtering_torch import Content_Based_filtering
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

class Hybrid_model(nn.Module):
    def __init__(self, ncf:Neural_Collaborative_filtering, content_filter:Content_Based_filtering, input_dim, binary=False) -> None:
        super(Hybrid_model, self).__init__()
        self.ncf = ncf
        self.content_recommender = content_filter

        self.l1 = nn.Linear(input_dim*2, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 1)

        self.binary = binary

    def forward(self, user_feat, item_feat, user_params, item_params, expand, weights):
        # ncf output
        V_ncf = self.ncf.raw_predict(user_params, item_params, expand=expand)
        V_ncf = F.normalize(V_ncf, p=2, dim=1)

        # content recommender output
        V_content_recommender = self.content_recommender.predict(user_feat, item_feat, expand=expand)
        V_content_recommender = F.normalize(V_content_recommender, p=2, dim=1)

        # final prediction
        ncf_weight, c_weight = weights
        x = torch.concat([V_ncf*ncf_weight, V_content_recommender*c_weight], dim=-1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        if self.binary :
            x = F.sigmoid(self.l4(x))
        else :
            x = self.l4(x)

        return x


class Hybrid_recommendation_system():
    def __init__(self, num_user:int, num_item:int, u_feat_dim:int, i_feat_dim:int, *, lr:float=0.001, l_d=20, binary=False) -> None:
        """
        Hybrid recommendation system based on the use of a Neural Collaborative filtering system and a content based recommender system using pytorch\n

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

        # initialize hyperparameters and model parameters
        self.epochs = 200
        self.learning_rate = lr
        self.weights = (0.5, 0.5)
        self.binary = binary

        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")

        self.create_model(u_feat_dim, i_feat_dim, l_d, binary)

        self.model_path = "hybrid_filtering_pytorch/models/hybrid"

    def create_model(self, u_dim, i_dim, latent_dimension, binary):
        # preparing the ncf algorithm
        collab_feature_dimension = 128
        self.ncf = Neural_Collaborative_filtering(self.num_user, self.num_item, collab_feature_dimension, output_dim=latent_dimension)

        # prepare the content based algorithm
        model_dim = 64
        self.content_filtering = Content_Based_filtering(u_dim, i_dim, model_dim, output_dim=latent_dimension)
        
        self.hybrid_model = Hybrid_model(self.ncf, self.content_filtering, latent_dimension, binary).to(self.device)

    def prediction(self, u_features, i_features, u_params=None, i_params=None, indexes=None, weights:tuple=None, expand=False) -> torch.Tensor:
        """
        Make a prediction of the affinity of a user to a certain item using movie and item features and parameters\n

        if value where normalized during training, make sure to apply a reverse process to the prediction in order to get appropriate value\n

        return >>> prediction shape(batch_size, 1)
        """
        weights = weights if weights else self.weights

        if indexes :
            u_params, i_params = self.ncf.get_params_from_idx(indexes) 

        rating = self.hybrid_model(u_features, i_features, u_params, i_params, expand, weights)

        return rating
    
    def __call__(self, u_features, i_features, u_params=None, i_params=None, indexes=None, weights:tuple=None, expand=False) -> np.ndarray:
        output = self.prediction(u_features, i_features, u_params, i_params, indexes, weights, expand)
        return output.detach().cpu().numpy()
        
    # train the models
    def train(self, u_features, i_features, param_idx ,ratings, rating_mask, expand=False, epochs=0, verbose=False) :
        """
        Compute loss with respect to combine recommendation of both recommender system
        and then backpropagate the gradient with respect to all learables variables in the overall system
        """
        ratings = torch.tensor(ratings.reshape(-1, 1), dtype=torch.float32, device=self.device)
        rating_mask = torch.tensor(rating_mask.reshape(-1, 1), dtype=torch.bool, device=self.device)

        u_features = torch.tensor(u_features, dtype=torch.float32, device=self.device)
        i_features = torch.tensor(i_features, dtype=torch.float32, device=self.device)

        previous_loss = 0
        self.num_item = i_features.shape[0]
        self.num_user = u_features.shape[0]

        # getting the user and item parameters
        u_param, i_param = self.ncf.get_params_from_idx(param_idx)
        u_param = nn.Parameter(torch.tensor(u_param, dtype=torch.float32, device=self.device), requires_grad=True)
        i_param = nn.Parameter(torch.tensor(i_param, dtype=torch.float32, device=self.device), requires_grad=True)

        def loss_func(u_feat, i_feat, ratings, rating_mask):
            ncf_weight, content_r_weight = self.weights 

            output = self.prediction(u_feat, i_feat, u_params=u_param, i_params=i_param, weights=(ncf_weight, content_r_weight), expand=expand)

            # filtering the empty values
            filtered_output = output[rating_mask]
            filtered_rating = ratings[rating_mask]

            if self.binary :
                criterion = nn.BCELoss()

                loss= criterion(filtered_output, filtered_rating)
                return loss
            
            else :
                squared_error = (filtered_output - filtered_rating)**2
                return torch.sum(squared_error)

        # initializing new optimizer cause we might be using new params for the ncf
        optimizer = torch.optim.Adam([
            {"params" : self.hybrid_model.parameters()},
            {"params" : u_param},
            {"params" : i_param},
            {"params" : self.hybrid_model.content_recommender.model.parameters()}
        ], lr=self.learning_rate)            

        # training loop
        training_epochs = epochs if epochs else self.epochs
        for iteration in range(training_epochs) :
            loss = loss_func(u_features, i_features, ratings, rating_mask)

            # gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and iteration % 20 == 0 :
                print("for epoch ", iteration, " the loss : ", loss)

            if iteration > 200 :
                if previous_loss < loss :
                    print("early stop at iteration : ", iteration)
                    break

                previous_loss = loss

        # setting the new user params at the indexes
        self.ncf.set_params_at_idx(u_param.detach().cpu().numpy(), i_param.detach().cpu().numpy(), param_idx)
        
        return loss


    def predict_latent_vec(self, features, isUser=True):
        """Make a prediction a parameters using the content filter parameters vector and an output model"""
        if isUser :
            latent_features = self.content_filtering.user_model(features)
            return latent_features
        
        else :
            latent_features = self.content_filtering.item_model(features)
            return latent_features
        
    def save_model(self, model_set=""):
        self.ncf.save_model(model_set=model_set)
        self.content_filtering.save_model(model_set=model_set)

        torch.save(self.hybrid_model.state_dict(), f"{self.model_path}/hybrid_model{model_set}.pth")
        print("all model saved")

    def load_model(self, model_set=""):
        self.hybrid_model.load_state_dict(torch.load(f"{self.model_path}/hybrid_model{model_set}.pth"))

        self.ncf.load_model(model_set=model_set)
        self.content_filtering.load_model(model_set=model_set)

        self.hybrid_model.ncf = self.ncf
        self.hybrid_model.content_recommender = self.content_filtering

        # loading info variables
        self.num_user = self.ncf._user_params.shape[0]
        self.num_item = self.ncf._item_params.shape[0]

        print("all models loaded")


# test accuracy