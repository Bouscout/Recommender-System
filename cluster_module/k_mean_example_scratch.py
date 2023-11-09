# tentative of implementing K_mean clustering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
np.random.seed(5)


image_path = "C:/Users/boudi/Downloads/tiny_back.jpg"
image = img.imread(image_path)
original_shape = image.shape
K = 8
# image shape
image = image.reshape(original_shape[0] * original_shape[1], 3)
print(image.shape)


def random_centroid():
    random_index = np.random.permutation(len(image))
    random_centroid = image[random_index[:K]]
    return random_centroid

def closest_centroids(x, centroids):
    distances_raw = (image[:, None, :] - centroids[None, :, :])**2
    distances = np.sum(distances_raw, axis=-1)

    index = np.argmin(distances, -1)

    return index


def change_centroids(x, cluster_index) :
    cluster_array = [[] for _ in range(K)]
    new_centroids = np.zeros((K, image.shape[-1]))
    
    # for i in range(K) :
                


    def new_centroid(points):
        return np.mean(points, axis=0)


    # for i in range(K) :
    #     if i in cluster_dict :
    #         cluster_pixel = cluster_dict[i]
    #         new_value = new_centroid(cluster_pixel)
    #         new_centroids[i] = new_value

    return new_centroids





# determine the cluster centroid points

def K_mean_iteration(x, centroids, iter=10):
    for _ in range(iter) :
        indices = closest_centroids(x, centroids)
        centroids = change_centroids(x, indices)

        print("iteration : ", _ + 1)

    return centroids, indices


centroids = random_centroid()
centroids, index = K_mean_iteration(image, centroids, iter=10)

max_pixel = np.max(centroids, axis=-1)
min_pixel = np.min(centroids, axis=-1)

# assign each pixel to its centroid
img_compressed = centroids[index, :]

img_compressed = img_compressed.reshape(original_shape)
img_compressed = img_compressed.astype(np.uint8)

image = image.reshape(original_shape)


plt.imshow(image)
plt.show()
input("")

plt.imshow(img_compressed)
plt.show()
input("")


