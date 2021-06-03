

import numpy as np 
import random
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt



def get_border_coords(border_data,tumor_labels):
  """
  Gets the x and y coordinates from the border_data for images with 15+ coordinates

  ---------------

  Parameters
  border_data: list that contains pairs of x and y coordinates
  tumor_labels: list of labels (1-3) indicating the class of the tumor

  ---------------

  returns lists x and y which contain the x and y coordinates respectively and also a list of labels for the subset of data with 15+ coordinates
  """

  # declaring 3 lists which will contain the x/y coordinates and the labels
  x = []
  y = []
  labels = []

  # looping through the border data 
  for i in np.arange(len(border_data)):

    # getting the ith label
    label = tumor_labels[i]

    # if we have less than 15 coordinate pairs, then we move on to the next iteration
    if border_data[i].shape[0] < 30:
      continue

    else:

      # x coords represents the even terms of the border data while y_coords represents the odd terms
      x_coords = list(border_data[i][::2])
      y_coords = list(border_data[i][1::2])

      # getting a random subset of 15 coordinate pairs
      x_sub, y_sub = zip(*random.sample(list(zip(x_coords, y_coords)), 15))
    
      # appending the lists with respective values
      x.append(x_sub)
      y.append(y_sub)
      labels.append(label)

  return x,y,labels


def create_linear_models(tumor_labels, x, y):
  """
  Creates two linear regression models that predict the x and y coordinates from the labels

  ---------------

  Parameters
  tumor_labels: an array of tumor labels (1-3)
  x: array of x coordinates for the tumors
  y: array of y coordinates for the tumors

  ---------------

  returns two linear regression models for x and y coordinates
  """
  # creating linear regression model for the x coordinates
  LR_x = LinearRegression().fit(tumor_labels, x)

  # creating linear regression model for the y coordinates 
  LR_y = LinearRegression().fit(tumor_labels, y)

  return LR_x, LR_y




def plot_classified_images_with_tumor_position(images, true_labels,labels_pred, label_encoder, x_points, y_points, true_tumor_labels, input = "correct"):
  """
  Plots the images that were correctly or incorrectly classified by the CNN model with tumor positions labeled on the plot

  ---------------

  Parameters
  images: testing images that the model has not seen
  true_labels: actual labels in the testing set
  labels_pred: predicted labels from the model
  label_encoder: dictionary that maps an integer to each tumor/non-tumor class
  x_points: x coordinates for the tumor 
  y_points: y coordinates for the tumor
  true_tumor_labels: list of labels containing only tumorous images
  input: string that determines if we want correctly classified or incorrectly classified images
  ---------------

  returns a matplotlib figure with 25 correctly classified or incorrectly classified MRI images with tumor position indicated
  """
  
  # creating a figure 
  figure = plt.figure(figsize=(20,20))

  # getting list of the keys and values from the label encoder
  key_list = list(label_encoder.keys())
  val_list = list(label_encoder.values())


  # checks if the user wants to plot correctly or incorrectly classified images and gets the respective list of labels
  if input == "correct":
    labels = np.arange(len(true_labels))[true_labels == labels_pred]
  elif input == "incorrect":
    labels = np.arange(len(true_labels))[true_labels != labels_pred]
  
  # shuffling the labels so that we get various tumor types plotted
  np.random.shuffle(labels)

  # calling create_linear_models to get predictions of the x and y coordinates
  LR_x, LR_y = create_linear_models(true_tumor_labels, x_points, y_points)

  # for loop that will plot 25 images
  for i in range(25):

    # getting the index of the ith correctly or incorrectly classified image
    idx = labels[i]

    # creating subplots and making aesthetic adjustments
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # if the true label is healthy, then plot the image along with its respective prediction
    if label_encoder.get(true_labels[idx]) == "Healthy":
      plt.imshow(images[idx].reshape(images.shape[1],images.shape[1]))
      plt.xlabel("Predicted: " + str(label_encoder.get(labels_pred[idx])) +  "\nActual: " + str(label_encoder.get(true_labels[idx])))

    else:

      # getting the true label
      position = val_list.index(label_encoder.get(true_labels[idx]))
      label =  key_list[position]
      
      # getting the x and y coordinates from the model for a particular label
      x_coords = LR_x.predict(np.array([[label]]))
      y_coords = LR_y.predict(np.array([[label]]))

      # putting the x and y coordinates in a tuple and appending the tuples to a list
      coords = []
      for i in np.arange(x_coords.shape[1]):
        coords.append((x_coords[0][i], y_coords[0][i]))

      # plotting the MRI image
      plt.imshow(images[idx].reshape(images.shape[1],images.shape[1]))

      # plotting the 'actual' tumor location as a black point
      plt.scatter(*zip(*coords), c = 'black')

      # labeling the x axis with predicted vs actual label
      plt.xlabel("Predicted: " + str(label_encoder.get(labels_pred[idx])) +  "\nActual: " + str(label_encoder.get(true_labels[idx])))

  return figure

