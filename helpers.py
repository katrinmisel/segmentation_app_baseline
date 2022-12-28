from labels import *
import os
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2

def id_to_cat(id):
    id2cat = {label.id : label for label in labels}
    return id2cat[id].categoryId

coarsify_mask = np.vectorize(id_to_cat)

def coarsify_ohe(mask):
    coarse_mask = coarsify_mask(mask)
    coarse_mask_ohe = tf.one_hot(coarse_mask, 8)
    #coarse_mask_ohe = np.squeeze(coarse_mask_ohe)
    return coarse_mask_ohe

def getPaths(dir):

  paths = []

  for file in os.listdir(dir):
    if file != "desktop.ini":
        full_path = dir + '/' + file
        paths.append(full_path)

  return sorted(paths)

# transform = A.Compose([
#     A.Rotate(limit=7, p=0.5),
#     A.HorizontalFlip(p=1),
#     A.RandomBrightnessContrast(p=0.5),
#     A.GaussNoise(var_limit=(0, 255), p=0.2), 
#     A.MotionBlur(blur_limit=33, p=0.3)
# ])

transform = A.Compose([
    A.HorizontalFlip(p=1), # Randomly flip the image horizontally and vertically
    A.Rotate(limit=5, p=0.5), # Randomly rotate the image by up to 45 degrees
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # Randomly adjust brightness and contrast
    A.GaussNoise(var_limit=(10.0, 50.0), mean=-50, p=0.2),
    A.CLAHE(clip_limit=6.0, tile_grid_size=(10, 10), p=0.3)
])

def augment_batch(batch_x, batch_y_ohe):

    batch_x_aug = []
    batch_y_aug = []

    for i in range (0, len(batch_x)):
        
        transformed_img_mask = transform(image=batch_x[i], mask=batch_y_ohe[i])

        img_aug = transformed_img_mask['image']
        mask_aug = transformed_img_mask['mask']

        batch_x_aug.append(img_aug)
        batch_y_aug.append(mask_aug)

    return batch_x_aug, batch_y_aug
