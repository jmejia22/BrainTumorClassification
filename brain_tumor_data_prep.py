#! pip install opencv-python
#! pip install keras

import os
import numpy as np 
from pymatreader import read_mat
import random
from cv2 import cv2
from keras.preprocessing.image import ImageDataGenerator


def get_tumor_images(url):
  """
  Retrieves the raw data needed for the CNN model, including the images, labels, 
  border, and mask for the 3 different types of tumor 

  ---------------

  Parameters
  url: the path url containing the location of the tumor images 

  ---------------

  returns lists of images, labels, and coordinates for each respective tumor 
  """

  # creating lists for meningioma data
  meningioma_images = []
  meningioma_labels = []
  meningioma_border = []
  meningioma_mask = []

  # creating lists for glioma data
  glioma_images = []
  glioma_labels = []
  glioma_border = []
  glioma_mask = []

  # creating lists for pituitary data
  pituitary_images = []
  pituitary_labels = []
  pituitary_border = []
  pituitary_mask = []


  # looping through tumorous data
  for i in os.listdir(url):
  
    # getting the data for a particular MRI and translating the .mat files using the read_mat function
    data = read_mat(os.path.join(url, i))

    # getting the image and label using indexing
    image = data['cjdata']['image']
    label = data['cjdata']['label']
    border = data['cjdata']['tumorBorder']
    mask = data['cjdata']['tumorMask']
    resized_border = border / 6.4

    # specifying what dimensions we want the image to be
    dimensions = (80,80)

    # resizing the image
    resized_image = cv2.resize(image, dimensions)
    resized_mask = cv2.resize(mask, dimensions)

    # checking the label of the image and appending each respective list accordingly 
    if label == 1:
      meningioma_images.append(resized_image)
      meningioma_labels.append(label)
      meningioma_border.append(resized_border)
      meningioma_mask.append(resized_mask)

    elif label == 2:
      glioma_images.append(resized_image)
      glioma_labels.append(label)
      glioma_border.append(resized_border)
      glioma_mask.append(resized_mask)

    else:
      pituitary_images.append(resized_image)
      pituitary_labels.append(label)
      pituitary_border.append(resized_border)
      pituitary_mask.append(resized_mask)

  return meningioma_images, meningioma_labels, meningioma_border, meningioma_mask, \
         glioma_images, glioma_labels, glioma_border, glioma_mask, \
         pituitary_images, pituitary_labels, pituitary_border, pituitary_mask
  


def get_healthy_images(url):
  """
  Retrieves the raw data needed for the CNN model, including the images and labels for the non-tumorous data 

  ---------------

  Parameters
  url: the path url containing the location of the healthy images 

  ---------------

  returns lists of images and labels for healthy images 
  """

  # creating lists for healthy data 
  no_tumor_images = []
  no_tumor_labels = []
  no_tumor_masks = []

  # looping through the healthy images 
  for i in os.listdir(url):

    # converting the image from jpg to an array
    image = cv2.imread(os.path.join(url,i))

    # if the shape of the image is not square, then we skip it
    if image.shape[0] != image.shape[1]:
      continue
  
  # converting the image to grayscale and resizing it, also appending the lists with the modified image and the label (4)
    else:
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dimensions = (80,80)
      resized_gray_image = cv2.resize(gray_image, dimensions)
      no_tumor_images.append(resized_gray_image)
      no_tumor_labels.append(4)
      no_tumor_masks.append(np.zeros(resized_gray_image.shape))

  return no_tumor_images, no_tumor_labels, no_tumor_masks


def image_augmentation(healthy_image_list):
  """
  Creates new non-tumorous images out of existing images by changing the image features.  

  ---------------

  Parameters
  healthy_image_list: list of images (arrays) which contains the non-tumorous images 

  ---------------

  returns lists of altered images 
  """

  # creating lists which will contain modified images
  zoomed_images = []
  rotated_images = []
  flipped_images = []
  bright_images = []


# looping from 0 to number of non-tumorous images (150)
  for i in np.arange(len(healthy_image_list)):

    # getting the ith original healthy image
    data = healthy_image_list[i]
 
    # reshaping the image
    data = np.reshape(data, (1,data.shape[0],data.shape[0],1))

    # generating different images
    datagen_zoom = ImageDataGenerator(zoom_range = [0.5,1.0])
    datagen_rotation = ImageDataGenerator(rotation_range = 90)
    datagen_flip = ImageDataGenerator(horizontal_flip = True)
    datagen_brightness = ImageDataGenerator(brightness_range=[0.2,0.6])

    # iterators for each modified image which will contain the modified image
    it_zoom = datagen_zoom.flow(data, batch_size=1)
    it_rotation = datagen_rotation.flow(data, batch_size = 1)
    it_flip = datagen_flip.flow(data, batch_size = 1)
    it_brightness = datagen_brightness.flow(data, batch_size = 1)


    # calling next on the iterator to get the modified image 
    batch_zoom = it_zoom.next()
    batch_rotation = it_rotation.next()
    batch_flip = it_flip.next()
    batch_brightness = it_brightness.next()

    # reshaping the modified images to match the shape of the original image
    image_zoom  = np.reshape(batch_zoom[0].astype('uint8'), (data.shape[1],data.shape[1]))
    image_rotation = np.reshape(batch_rotation[0].astype('uint8'), (data.shape[1], data.shape[1]))
    image_flip = np.reshape(batch_flip[0].astype('uint8'), (data.shape[1], data.shape[1]))
    image_brightness = np.reshape(batch_brightness[0].astype('uint8'), (data.shape[1], data.shape[1]))

    # appending the lists with the modified image
    zoomed_images.append(image_zoom)
    rotated_images.append(image_rotation)
    flipped_images.append(image_flip)
    bright_images.append(image_brightness)

  return zoomed_images, rotated_images, flipped_images, bright_images


def random_sample_healthy_images(original_images, zoomed_images, rotated_images, flipped_images, bright_images, n):
  """
  Takes a random sample of n elements from each of the lists of images with altered features including the original images

  ---------------

  Parameters
  original_images: list containing original non-tumorous images
  zoomed_images: list containing original non-tumorous images zoomed in 
  rotated_images: list containing original non-tumorous images rotated
  flipped_images: list containing original non-tumorous images flipped about horizontal axis
  bright_images: list containing original non-tumorous images brightened/darkened
  n: number of images we want sampled from each of the above lists

  ---------------

  returns list of images with n samples from each altered & original image list as well as a list of labels
  """
  no_tumor_images_new = []
  no_tumor_labels = []

  for i in np.arange(710):
    no_tumor_labels.append(4)

  # getting random sample from each modified image list
  random_original = random.sample(original_images,n)
  random_zoom = random.sample(zoomed_images, n)
  random_rotated = random.sample(rotated_images, n)
  random_flipped = random.sample(flipped_images, n)
  random_bright = random.sample(bright_images, n)

  # appending these images to our new list of non-tumorous images
  no_tumor_images_new = random_original + random_zoom + random_rotated + random_flipped + random_bright

  return no_tumor_images_new, no_tumor_labels


def get_predictor_target_data(meningioma_images, meningioma_labels, glioma_images, glioma_labels, pituitary_images, pituitary_labels, no_tumor_images, no_tumor_labels):
  """
  Combines all of the tumorous and non-tumorous data and separates them into predictor (images) and target (labels) data

  ---------------

  Parameters
  meningioma_images: list containing meningioma tumor images
  meningioma_labels: list containing meningioma tumor labels
  glioma_images: list containing glioma tumor images
  glioma_labels: list containing glioma tumor labels
  pituitary_images: list containing pituitary tumor images 
  pituitary_labels: list containing pituitary tumor labels
  no_tumor_images: list containing non-tumorous images
  no_tumor_labels: list containing non-tumorous labels

  ---------------

  returns list of images containing all tumor classes and list of lables containing all tumor classes 
  """

  # getting subset of 709 images and labels for glioma and pituitary tumors
  glioma_images_subset, glioma_labels_subset = zip(*random.sample(list(zip(glioma_images, glioma_labels)), 709))
  pituitary_images_subset, pituitary_labels_subset = zip(*random.sample(list(zip(pituitary_images, pituitary_labels)), 709))

  # converting all subsets to a list
  glioma_images_subset = list(glioma_images_subset)
  glioma_labels_subset = list(glioma_labels_subset)

  pituitary_images_subset = list(pituitary_images_subset)
  pituitary_labels_subset = list(pituitary_labels_subset)

  # appending all images to a list called X_data and all labels to a list called y_data
  X_data = meningioma_images + glioma_images_subset + pituitary_images_subset + no_tumor_images
  y_data = meningioma_labels + glioma_labels_subset + pituitary_labels_subset + no_tumor_labels

  return X_data, y_data


def get_border_data(meningioma_labels, meningioma_border, glioma_labels, glioma_border, pituitary_labels, pituitary_border):
  """
  Combines all of the tumorous border and label data for each tumor class into two respective lists

  ---------------

  Parameters
  meningioma_labels: list containing meningioma tumor labels
  meningioma_border: list containing meningioma tumor coordinates
  glioma_labels: list containing glioma tumor labels
  glioma_border: list containing glioma tumor coordinates 
  pituitary_labels: list containing pituitary tumor labels
  pituitary_border: list containing pituitary tumor coordinates

  ---------------

  returns list of coordinates containing all tumorous classes and list of lables containing all tumorous classes 
  """

  # getting a random subset of 709 tumor coordinates and their respective labels 
  glioma_labels_subset, glioma_border_subset = zip(*random.sample(list(zip(glioma_labels, glioma_border)), 709))
  pituitary_labels_subset, pituitary_border_subset = zip(*random.sample(list(zip(pituitary_labels, pituitary_border)), 709))

  # converting the subsets to a list
  glioma_labels_subset = list(glioma_labels_subset)
  glioma_border_subset = list(glioma_border_subset)

  pituitary_labels_subset = list(pituitary_labels_subset)
  pituitary_border_subset = list(pituitary_border_subset)

  # appending all labels to a list called tumor_labels and all coordinates to a list called border_data 
  tumor_labels = meningioma_labels + glioma_labels_subset + pituitary_labels_subset
  border_data = meningioma_border + glioma_border_subset + pituitary_border_subset

  return tumor_labels, border_data

def get_mask_data(meningioma_images, meningioma_mask, glioma_images, glioma_mask, pituitary_images, pituitary_mask, no_tumor_images, no_tumor_mask):
  """
  Combines all of the tumorous border and label data and separates them into two respective lists

  ---------------

  Parameters
  meningioma_images: list containing meningioma tumor images
  meningioma_masks: list containing meningioma tumor masks
  glioma_images: list containing glioma tumor images
  glioma_masks: list containing glioma tumor masks
  pituitary_images: list containing pituitary tumor images 
  pituitary_masks: list containing pituitary tumor masks
  no_tumor_images: list containing non-tumorous images
  no_tumor_masks: list containing non-tumorous masks
  ---------------

  returns list of images containing all tumorous classes and list of masks containing all tumorous classes 
  """

  # getting a random subset of 709 tumor coordinates and their respective labels 
  glioma_images_subset, glioma_mask_subset = zip(*random.sample(list(zip(glioma_images, glioma_mask)), 709))
  pituitary_images_subset, pituitary_mask_subset = zip(*random.sample(list(zip(pituitary_images, pituitary_mask)), 709))

  # converting the subsets to a list
  glioma_images_subset = list(glioma_images_subset)
  glioma_mask_subset = list(glioma_mask_subset)

  pituitary_images_subset = list(pituitary_images_subset)
  pituitary_mask_subset = list(pituitary_mask_subset)

  # appending all labels to a list called tumor_labels and all coordinates to a list called border_data 
  X_images = meningioma_images + glioma_images_subset + pituitary_images_subset + no_tumor_images
  y_mask_data = meningioma_mask + glioma_mask_subset + pituitary_mask_subset + no_tumor_mask

  return X_images, y_mask_data



def list_to_array(lst):
  """
  Converts a list to an array

  ---------------

  Parameters
  lst: an arbitrary list

  ---------------

  returns an array
  """
  arr = np.array(lst)
  
  return arr


def _3d_to_4d(arr):
  """
  Converts a 3D array into a 4D array

  ---------------

  Parameters
  arr: a 3D array

  ---------------

  returns a 4D array
  """
  arr = np.reshape(arr, (arr.shape[0], arr.shape[1], arr.shape[1], 1))
  
  return arr




