# Recommender-System
Recommendation system including module like implementation of collaborative filtering, content based filtering and hybrid recommendation system laveraging a weighted application of both previous systems.

## Requirements
```
numpy==1.23.5
pandas==2.1.2
python-dotenv==1.0.0
scikit-learn==1.3.2
tensorflow==2.12.0
torch==2.1.0+cu121
```

## Neural Collaborative Filtering (NCF)
This class implements a collaborative filtering system using a neural network for predicting user-item affinity.
* Parameters :
  * num_user: The number of users.
  * num_item: The number of items.
  * x_dim: The dimension of user and item vectors.
  * learning_rate: The learning rate for optimization (default: 0.0001).
  * _lambda_: The regularization parameter (default: 1.5).
  * output_dim: The number of output dimensions (default: 1).
* Methods :
  * loss_func(self, w, x, y_true, r): Calculate the loss function for the neural collaborative filtering model.
  * normalize_label(self, all_ratings, rating_mask, num_user, num_movie): Normalize ratings by subtracting the mean.
  * train(self, rating, rating_mask, indexes=None, epochs=200, verbose=False, normalize=True): Train the model on a given batch of data.
  * get_params(self): Get user and item parameters.
  * set_params(self, params) : set the parameters to the new values provided.
  * predict(self, user_param=None, item_param=None, indexes=None, expand=False, normalize=True): Make predictions using the trained model.
  * raw_predict(self, user_param=None, item_param=None, indexes=None, flatten=True, expand=False): Make raw predictions using the trained model.
  * save_model(self): Save the model and parameters to files.
  * load_model(self): Load the model and parameters from files.
* Example :
  Here is an example on how to use it for equal number of items and users :
  ```python
  from hybrid_filtering.NCF import Neural_Collaborative_filtering
  
  model = Neural_Collaborative_filtering(num_user, num_item, x_dim)
  
  # Train the model
  model.train(rating_data, rating_mask, epochs=200)
  
  # Make predictions
  predictions = model.predict(user_param, item_param)
  
  # Save the model
  model.save_model()
  ```
  If you want to make a prediction for every item-user pairs possible, set the expand parameter to true
  ```python
  from hybrid_filtering.NCF import Neural_Collaborative_filtering
  
  model = Neural_Collaborative_filtering(num_user, num_item, x_dim)

  predictions = model.predict(user_param, item_param, expand=True)
  ```

  you can also specify specific indexes by providing an index tuple for user and item
  ```python
  user_index = np.arange(0,  num_user_training)
  item_index = np.arange(0, num_item_training)

  model.train(rating_data, rating_mask, indexes=(user_index, item_index))
  ```
more informations on specific implementation details in the documentations (code comments)
## Content Based filtering
This class implements a content based filtering system using a neural network for predicting user-item affinity between a user and item latent vectors.
* Parameters :
  * Xu: User input vector dimension.
  * Xi: Item input vector dimension.
  * V_dim: Latent vector dimension representing the latent variable of user and item features.
  * lr: Learning rate for optimization.
  * special_dimension (optional): Dimension for special cases
* Methods :
  * predict(self, user_input, item_input, flatten, expand): make a of user-item affinity.
  * train (self, user_input, item_input, rating, rating_mask): train the model based on the ratings passed.
  * save_model, load_model : save or load the model
*Example :
Here is basic usage of the model, note that you can also expand the predictions to every user-item pair by setting expand=True.
```python
from hybrid_filtering.content_based_filtering import Content_Based_filtering

model = Neural_Collaborative_filtering(num_user, num_item, x_dim)

# Train the model
model.train(user_feat, item_feat, rating, rating_mask, epochs=200)

# Make predictions
predictions = model.predict(user_feat, item_feat, expand=True)

# Save the model
model.save_model()
```
more informations on specific implementation details in the documentations (code comments)
## Hybrid Filtering
This class implements a combinaison of both collaborative system described above, it then laverages both of their predictions by representing them using a latent vector of dimensions "l_d" and then apply a given scalar weights to both of those latent vectors before concatenating them and passing them to a last model which will make the final prediction.

Example :
```python
import numpy as np
from hybrid_filtering.hybrid_recommendation_system import Hybrid_recommendation_system

# Create an instance of Hybrid_recommendation_system
hybrid_system = Hybrid_recommendation_system(num_user=100, num_item=150, u_feat_dim=32, i_feat_dim=128, lr=0.001, l_d=64)

# Generate sample data (replace with your data)
user_features = np.random.rand(100, 50)
item_features = np.random.rand(150, 50)
ratings = np.random.rand(100, 150)
rating_mask = (ratings > 0).astype(int)
param_indexes = (np.random.randint(0, 100), np.random.randint(0, 150))

# Train the hybrid model
hybrid_system.weights = (0.5, 0.5)
hybrid_system.train(user_features, item_features, indexes=param_indexes, ratings, rating_mask, expand=True, epochs=200, verbose=True)

# Make predictions
predictions = hybrid_system.prediction(user_features, item_features, indexes=param_indexes, weights=(0.5, 0.5), expand=True)
print(predictions)
```
## Implementation In A Project
This recommendation system has already been implemented in a large project and those specific details can be found in the project "Buushido". Here is a schema for design of that implementation. 

We divide the problem into 4 different aspects : 
Embedding Retrieval,
Ranking,
Label Collection,
Cold Start Problem,

### Embedding Retrieval
This part represent the process of selecting a sample of potential relevant items using the user informations.
To address this problem, we laverage the latent variable of the trained model to classify the user and items into cluster using a Kmeans clustering technique.
We use both the user model and item model present in the content filtering system of the hybrid system, we predict the latent vectors of all items and users and then classify them into user clusters and item clusters. We then proceed to identify which user cluster is attracted to which item cluster and set up a top three cluster for every user cluster.

Therefore in the retrieval situation, you would need to find the user cluster from the informations you had access to, you will then retrieve the stored user item cluster top 3 and use those embeddings for the next step.

### Ranking
Once we have access to the embeddings and the user informations, you simple perform a prediction using the model for every item in the embedding with respect to the user.
We will then sort those items according to their ratings and then recommend a number of items n to the user based on the sorted items.

### Label Collection
This part concerns retrieving the result of those predictions. It will totally depend on the technique used to label the interactions that the the user had with the recommended show. We will need to ensure that we have access to the informations used to make that predictions and the label made to that predictions. Those details will enable us to perform additional training regardless of the method of labelisation employed.

### Cold Start Problem
We might run into a situation where we don't have enough user or item data to find the patern in the model. Addressing this problem would depend on the situation and the type of problem so I will only mention my solution as for the Buushido project.

The project concerned anime data, so with access to myanimelist public api data, I could access various statistics about each item as much as various features data and even some recommendations based anime ressemblance. With that amount of data, it was possible to create using some probability setting a database representing a number user n and their interactions with a number of item m. From that database, it was possible to derive some user and item cluster and to create the whole system on that basis.

Finally to test the efficacity of that system, I conducted a survey with some users of my website, I would be provided some informations about the items they appreciated and would try to recommend additional items, the rate of satisfaction was over 40% which constituated a solid basis for starting the implementation.

### Buushido Implementation details
You can find more details on how I specifically implemented these components in the buushido project by visiting the documentation of the project.
