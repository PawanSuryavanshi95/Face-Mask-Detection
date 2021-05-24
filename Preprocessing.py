import cv2
import pickle
import numpy as np
import os
import random

random.seed(0)

directory = r"C:\Users\pawan\PycharmProjects\Face Mask\self-built-masked-face-recognition-dataset\AFDB_face_dataset"
directory2 = r"C:\Users\pawan\PycharmProjects\Face Mask\self-built-masked-face-recognition-dataset\AFDB_masked_face_dataset"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            images.append(img)
    return images

def collect_data(directory):
    sub_dir = os.listdir(directory)
    images = []
    for sub in sub_dir[0:1]:
        print(sub)
        images.extend(load_images_from_folder(os.path.join(directory, sub)))
    return images

def flat(images):
    images = np.array(images)
    flat_images = []
    for img in images:
        flat_images.append(img.flatten())
    return flat_images

def save(dic):
    with open('data', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

images = collect_data(directory)
images2 = collect_data(directory2)

L = len(images2)

# Selecting 'L' random images from 'images' list
images_sampled = random.sample(images, L)

# Flattening Images
flat1 = flat(images_sampled)
flat2 = flat(images2)

samples = flat1 + flat2

# Creating Labels

label1 = [1 for x in range(L)]
label2 = [0 for x in range(L)]
labels = label1+label2

# Final Data

data = {'samples':samples, 'labels':labels}

# saving data
save(data)

# Visualizing Images ( to cross check)

img = np.array(images_sampled[0])
print(img.shape)
cv2.imshow('Without Mask', img)

img = np.array(images2[0])
print(img.shape)
cv2.imshow('Masked', img)

# Closing
cv2.waitKey(0)

cv2.destroyAllWindows()