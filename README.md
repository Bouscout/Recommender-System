# Recommender-System
Recommendation system including module like implementation of collaborative filtering, content based filtering and hybrid recommendation system laveraging a weighted application of both previous systems.

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
