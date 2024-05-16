
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from tabulate import tabulate
import pandas as pd
from PIL import Image
import tifffile
import pandas as pd
import imageio
# def DownMappe(mappe):
#         Bands_1 = os.listdir(mappe)
#         l = []
#         filnavne = np.array([])
#         for file in Bands_1:
#             filnavn = str(file)
#             filnavne = np.append(filnavne,filnavn)
#             img = Image.open(os.path.join(mappe,file))
#             image_8bit = np.array(img)
            
#             image_8bit2 = np.reshape(image_8bit,(1,image_8bit.shape[0]*image_8bit.shape[1]))
#             l.append(image_8bit2)
#         size = image_8bit.shape
#         return np.transpose(np.vstack(l)), size

def DownMappe(mappe1, mappe2):
    Bands_1 = os.listdir(mappe1)
    l = []
    l2 = []
    filnavne = np.array([])
    for file in Bands_1:
        min_val = float('inf')
        max_val = float('-inf')
        filnavn = str(file)
        filnavne = np.append(filnavne,filnavn)
        img = Image.open(os.path.join(mappe1,file))
        img2 = Image.open(os.path.join(mappe2,file))
        arr = np.array(img)
        arr2 = np.array(img2)
        min_val = min(min_val, np.min(arr))
        min_val = min(min_val, np.min(arr2))
        max_val = max(max_val, np.max(arr))
        max_val = max(max_val, np.max(arr2))

        normalized_arr = ((arr - min_val) / (max_val - min_val)) * 255
        normalized_arr2 = ((arr2 - min_val) / (max_val - min_val)) * 255
        # plt.imsave(f'Date1{file}.png', normalized_arr, cmap="gray",vmin=0, vmax=255)
        # plt.imsave(f'Date2{file}.png', normalized_arr2, cmap="gray",vmin=0, vmax=255)

        # plt.subplot(1,2,1)
        # plt.imshow(normalized_arr)
        # plt.subplot(1,2,2)
        # plt.imshow(normalized_arr2)
        # plt.show()
        image_8bit2 = np.reshape(normalized_arr,(arr.shape[0]*arr.shape[1]))
        image_8bit22 = np.reshape(normalized_arr2,(arr.shape[0]*arr.shape[1]))
        l.append(image_8bit2)
        l2.append(image_8bit22)
    size = arr.shape
    return np.transpose(np.vstack(l)), np.transpose(np.stack(l2)), size


def PCAdiff(Billede1,Billede2,N,Download = False):
    pca = PCA(N)

    Billede1_pca = pca.fit_transform(Billede1)
    if Download == True:
        df = pd.DataFrame(np.transpose(pca.components_))
        df.to_excel("Data1LV.xlsx", index=False, engine='openpyxl')
    if N == 13:
        print(pca.explained_variance_)
        plt.rcParams["font.family"] ="serif"
        plt.bar([i for i in range(1,14)],pca.explained_variance_ratio_*100, label="Explained variance")
        plt.plot([i for i in range(0,14)], np.append(0,np.cumsum(pca.explained_variance_ratio_*100)),"orange",label="Accumulated Explained Variance")
        plt.xlabel("PC")
        plt.ylabel("Fraction of total variance [%]")
        plt.xlim(0,13)
        plt.xticks(np.arange(1, 14, step=1))
        plt.legend()
        plt.show()

    Billede2_pca = pca.fit_transform(Billede2)
    if Download == True:
        df = pd.DataFrame(np.transpose(pca.components_))
        df.to_excel("Data2LV.xlsx", index=False, engine='openpyxl')
    if N == 13:
        print(pca.explained_variance_)
        plt.rcParams["font.family"] ="serif"
        plt.bar([i for i in range(1,14)],pca.explained_variance_ratio_*100, label="Explained variance")
        plt.plot([i for i in range(0,14)], np.append(0,np.cumsum(pca.explained_variance_ratio_*100)),"orange",label="Accumulated Explained Variance")
        plt.xlabel("PC")
        plt.ylabel("Fraction of total variance [%]")
        plt.xlim(0,13)
        plt.xticks(np.arange(1, 14, step=1))
        plt.legend()
        plt.show()

    diff = np.abs(Billede1_pca-Billede2_pca)
    return diff



def DoPCA(by,N):
    mappe1 = fr"C:\Users\h4sor\OneDrive\Desktop\Uni\Kodnings projekter\Onera Satellite Change Detection dataset - Images\{by}\imgs_1_rect"
    mappe2 = fr"C:\Users\h4sor\OneDrive\Desktop\Uni\Kodnings projekter\Onera Satellite Change Detection dataset - Images\{by}\imgs_2_rect"

    og1 = Image.open(fr"C:\Users\h4sor\OneDrive\Desktop\Uni\Kodnings projekter\Onera Satellite Change Detection dataset - Images\{by}\pair\img1.png")
    og2 = Image.open(fr"C:\Users\h4sor\OneDrive\Desktop\Uni\Kodnings projekter\Onera Satellite Change Detection dataset - Images\{by}\pair\img2.png")

    Billede1,Billede2,size = DownMappe(mappe1,mappe2)

    # for i in range(13):
    #     plt.subplot(1, 2, 1)
    #     billede = Billede1[:,i]
    #     billede = billede.reshape(size[0], size[1], -1)
    #     plt.imshow(billede, cmap='gray')

    #     plt.subplot(1, 2, 2)
    #     billede = Billede2[:,i]
    #     billede = billede.reshape(size[0], size[1], -1)
        
    #     plt.imshow(billede, cmap='gray')
    #     plt.title(f"Band {i+1}")
    #     plt.show()
 

    change_reshaped = PCAdiff(Billede1,Billede2,N).reshape(size[0], size[1], -1)

    change_reshaped = np.clip(change_reshaped, 0, 255)
    # Theresh hold:
    # change_reshaped = np.where(change_reshaped>100, 255, 0)

    a = 3
    b = 5
    folder_path = 'LVPCS'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



    # Plot the change for each principal component
    plt.rcParams["font.family"] ="serif"
    for i in range(N):
        plt.subplot(a, b, i+1)
        # plt.imsave(f'LVPC{i+1}.png', change_reshaped[:, :, i], cmap="viridis",vmin=0, vmax=255)
        # os.rename(f'LVPC{i+1}.png', os.path.join(folder_path, f'LVPC{i+1}.png'))
        plt.imshow(change_reshaped[:, :, i], cmap='viridis',vmin=0, vmax=255 )
        plt.axis("off")
        plt.title(f'PC {i+1}')
    plt.show()


DoPCA("rio",13)