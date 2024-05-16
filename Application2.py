import os

# Set the environment variable to the number of physical cores you have
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Replace '4' with the number of physical cores

# Now you can run your Python script


import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt

def find_vector_set(diff_image, new_size):
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1

    mean_vec   = np.mean(vector_set, axis = 0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new):
    i = 2
    feature_vector_set = []

    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    return FVS

def clustering(FVS, components, new):
    kmeans = KMeans(n_clusters=components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)

    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    return least_index, change_map

def find_PCAKmeans(imagepath1, imagepath2):
    image1 = cv2.imread(imagepath1,cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(imagepath2, cv2.IMREAD_GRAYSCALE)

    new_size = np.asarray(image1.shape) / 5
    new_size = new_size.astype(int) * 5
    image1 = cv2.resize(image1, (new_size[1], new_size[0])).astype(np.int16)
    image2 = cv2.resize(image2, (new_size[1], new_size[0])).astype(np.int16)

    diff_image = np.abs(image1 - image2)
    cv2.imwrite('diff.jpg', diff_image)

    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_

    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0

    change_map = change_map.astype(np.uint8)
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map, kernel)
    cv2.imwrite("changemap.jpg", change_map)
    cv2.imwrite("cleanchangemap.jpg", cleanChangeMap)

    plt.rcParams["font.family"] ="serif"
    plt.subplot(1,3,1)
    plt.imshow(image1,cmap="gray")
    plt.title(r"$X_1$")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(image2,cmap="gray")
    plt.title(r"$X_2$")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(change_map,cmap="gray")
    plt.title("Change map")
    plt.axis("off")


    plt.show()



by = "rio"
a = fr"Onera Satellite Change Detection dataset - Images/{by}/pair/img1.png"
b = fr"Onera Satellite Change Detection dataset - Images/{by}/pair/img2.png"
find_PCAKmeans(a,b)