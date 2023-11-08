import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent_model(nn.Module):
    def __init__(self, user_model, item_model,input_shape, output_shape, binary=False) -> None:
        super(Latent_model, self).__init__()
        self.user_model = user_model
        self.item_model = item_model

        self.l1 = nn.Linear(input_shape*2, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, output_shape)

        self.binary = binary

    def forward(self, Xu, Xi):
        Vu = self.user_model(Xu)
        Vu = F.normalize(Vu, p=2, dim=1)

        Vi = self.item_model(Xi)
        Vi = F.normalize(Vi, p=2, dim=1)

        x = torch.concat([Vu, Vi], dim=-1)
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        if self.binary :
            x = F.sigmoid(x)
        else :
            x = self.l4(x)
        
        return x
        

class Content_Based_filtering():
    def __init__(self, Xu, Xi, V_dim, lr=0.0001, output_dim=1) -> None:
        """
        Implementation of Content Based filtering using pytorch which uses two neural network to represent two latent vector Vu and Vi
        of dimentions dim representing the latent variable of the different feature of the user and item input\n

        a matrix multiplication between those two latent vector represent the affinity or rating of the user toward that item
        required : \n
        user input vector dimension : Xu\n
        item input vector dimension : Xi\n
        latent vector dimension : Vm\n

        hyperparameters : learning_rate
        """
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")
    

        self.l_r = lr
        self.create_models(Xu, Xi, V_dim, output_dim)

        self.model_path = "hybrid_filtering_pytorch/models/content_filter"

    def create_models(self, Xu, Xi, Vm, output_dim):    
        # user model
        self.user_model = nn.Sequential(
            nn.Linear(Xu, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, Vm),
        )
        # item model
        self.item_model = nn.Sequential(
            nn.Linear(Xi, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, Vm),
        )

        self.model = Latent_model(self.user_model, self.item_model, Vm ,output_dim).to(self.device)

    def expand_dim_pair(self, Xu, Xi):
        num_user, num_item = Xu.shape[0], Xi.shape[0]

        Xu = Xu.unsqueeze(1).expand(-1, num_item, -1)
        Xi = Xi.unsqueeze(0).expand(num_user, -1, -1)

        return Xu, Xi

    def loss_func(self, Xu, Xi, y, r_mask) -> torch.Tensor :
        # concat each user for all movies
        expand_Xu, expand_Xi = self.expand_dim_pair(Xu, Xi)

        # flattening the extra dimensions
        Xu = torch.reshape(expand_Xu, (-1, Xu.shape[-1]))
        Xi = torch.reshape(expand_Xi, (-1, Xi.shape[-1]))

        # both model prediction
        y_pred = self.model(Xu, Xi)

        loss = (y_pred - y)**2

        # Apply the mask to the squared error to filter out non-rated items
        filtered_squared_error = loss[r_mask]

        return torch.sum(filtered_squared_error)
    
   

    def train(self, user_input, item_input, rating, rating_mask, epochs=200, verbose=False) :
        """
        Perform training over the given batch for a number of epochs
        """
        # we use the transpose version because I compute the loss from the perspective of each user
        # with respect to all items
        user_input = torch.tensor(user_input, dtype=torch.float32, device=self.device)
        item_input = torch.tensor(item_input, dtype=torch.float32, device=self.device)

        rating = torch.tensor(np.reshape(rating, (-1, 1)), dtype=torch.float32, device=self.device)
        rating_mask = torch.tensor(np.reshape(rating_mask, (-1, 1)), dtype=torch.bool, device=self.device)

        previous_loss = 10000

        optimzer = torch.optim.Adam(self.model.parameters(), lr=self.l_r)
        for iteration in range(epochs) :
            
            loss = self.loss_func(user_input, item_input, rating, rating_mask)

            # gradient step
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            if iteration % 20 == 0 and verbose :
                print("for epoch ", iteration, " the loss : ", loss)

            if iteration > 200 :
                if previous_loss < loss :
                    print("early stop at iteration : ", iteration)
                    break

                previous_loss = loss

        return loss

    def predict(self, user_input, item_input, expand=False):
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

            user_input = torch.reshape(user_input, (-1, user_input.shape[-1]))
            item_input = torch.reshape(item_input, (-1, item_input.shape[-1]))
            user_rank = 3
        
        elif user_input.shape[0] != item_input.shape[0]:
            raise ValueError(f"No matching number of user and item, user:{user_input.shape[0]} and item:{item_input.shape[0]}")

        rating = self.model(user_input, item_input)        
        return rating
    
    def save_model(self, model_set=""):
        torch.save(self.model.state_dict(), f"{self.model_path}/content_model{model_set}.pth")
        torch.save(self.model.user_model.state_dict(), f"{self.model_path}/user_model{model_set}.pth")
        torch.save(self.model.item_model.state_dict(), f"{self.model_path}/item_model{model_set}.pth")
        print("content model saved")

    def load_model(self, model_set=""):
        self.model.load_state_dict(torch.load(f"{self.model_path}/content_model{model_set}.pth"))
        self.model.user_model.load_state_dict(torch.load(f"{self.model_path}/user_model{model_set}.pth"))
        self.model.item_model.load_state_dict(torch.load(f"{self.model_path}/item_model{model_set}.pth"))
        print("content model loaded")