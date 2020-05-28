#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Updating X_train, Y_train and X_test, Y_test usnig 'Landset 8' Dataset

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
from PIL import Image

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH_R = '/content/drive/My Drive/Colab Notebooks/dataset/B4/train/'
TRAIN_PATH_G = '/content/drive/My Drive/Colab Notebooks/dataset/B3/train/'
TRAIN_PATH_B = '/content/drive/My Drive/Colab Notebooks/dataset/B2/train/'

TEST_PATH_R = '/content/drive/My Drive/Colab Notebooks/dataset/B4/test/'
TEST_PATH_G = '/content/drive/My Drive/Colab Notebooks/dataset/B3/test/'
TEST_PATH_B = '/content/drive/My Drive/Colab Notebooks/dataset/B2/test/'

X_train = np.zeros((350, IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((350, IMG_WIDTH, IMG_WIDTH, 1), dtype=np.float32)
img = np.zeros((IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

tr=np.zeros(350)
te=np.zeros(10)

for i in range(350):
       tr[i]=i;
for i in range(10):
       te[i]=i;

# for train images
        
for n, id_ in tqdm(enumerate(tr),total=350):
        red    = Image.open(TRAIN_PATH_R + str(int(id_)) + '.png').convert('L')
        green  = Image.open(TRAIN_PATH_G + str(int(id_)) + '.png').convert('L')
        blue   = Image.open(TRAIN_PATH_B + str(int(id_)) + '.png').convert('L')
       
        rgb = Image.merge("RGB",(red,green,blue))
        img_b = np.asarray(rgb) 
       
       #img_r = imread(TRAIN_PATH_R + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       #img_g = imread(TRAIN_PATH_G + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       #img_b = imread(TRAIN_PATH_B + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]

       #img_r = resize(img_r, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #img_g = resize(img_g, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)

        img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
        img_b=img_b/255.0
       #for i in range(256):
       #       img[i] = np.concatenate((img_r[i],img_g[i],img_b[i]), axis=1)
       
       X_train[n] = img_b
       
       #mask = Image.open('/content/drive/My Drive/Colab Notebooks/dataset/BQA/train/' + str(int(102+id_)) + '.png').convert('L')
        mask = imread('/content/drive/My Drive/Colab Notebooks/dataset/BQA/train/' + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
        mask1 = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #mask1 = np.asarray(mask)
       #mask1 = resize(mask1, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)

        Y_train[n] =mask1/255.0
        for io in range(256):
            for jo in range(256):
                if (Y_train[n][io][jo]>0.3):
                    Y_train[n][io][jo]=1
                else:
                    Y_train[n][io][jo]=0
        

#for test images
    

X_test = np.zeros((10, IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_test = np.zeros((10, IMG_WIDTH, IMG_WIDTH, 1), dtype=np.float32)
img = np.zeros((IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []


for n, id_ in tqdm(enumerate(te),total=10):
    red    = Image.open(TEST_PATH_R + str(170+int(id_)) + '.png').convert('L')
    green  = Image.open(TEST_PATH_G + str(170+int(id_)) + '.png').convert('L')
    blue   = Image.open(TEST_PATH_B + str(170+int(id_)) + '.png').convert('L')
       
    rgb = Image.merge("RGB",(red,green,blue))
    img_b = np.asarray(rgb)  

    img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
    img_b=img_b/255.0
       #for i in range(256):
       #       img[i] = np.concatenate((img_r[i],img_g[i],img_b[i]), axis=1)
       
    X_test[n] = img_b

    mask = imread('/content/drive/My Drive/Colab Notebooks/dataset/BQA/test/' + str(int(170+id_)) + '.png')[:,:,:IMG_CHANNELS]
    mask1 = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #mask1 = np.asarray(mask)
       #mask1 = resize(mask1, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)

    #Binarization using QA pixel values
    
    Y_test[n] =mask1/255.0
    for io in range(256):
        for jo in range(256):
            if (Y_test[n][io][jo]>0.3):
                 Y_test[n][io][jo]=1
            else:
                 Y_test[n][io][jo]=0

