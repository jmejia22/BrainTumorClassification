
import numpy as np 
from matplotlib import pyplot as plt
import itertools
from brain_tumor_position import create_linear_models




def accuracy_plot(history):
  """
  Plots the epoch on the x axis and train/validation accuracy on the y axis

  ---------------

  Parameters
  history: history object which records training metrics for each epoch.

  ---------------

  returns a matplotlib figure with epochs on the x axis and accuracy on the y axis  
  """
  
  # creating a figure
  figure = plt.figure(figsize=(12, 8))

  # plotting the training accuracy and validation accuracy
  plt.plot(history.history["accuracy"], label = "training")
  plt.plot(history.history["val_accuracy"], label = "validation")

  # labeling the plot
  plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
  plt.legend()
  plt.title("Plot of Training vs Validation Accuracy")

  return figure


def loss_plot(history):
  """
  Plots the epoch on the x axis and train/validation loss on the y axis

  ---------------

  Parameters
  history: history object which records training metrics for each epoch.

  ---------------

  returns a matplotlib figure with epochs on the x axis and loss on the y axis  
  """

  # creating a figure
  figure = plt.figure(figsize=(12, 8))

  # plotting the training loss and validation loss
  plt.plot(history.history["loss"], label = "training")
  plt.plot(history.history["val_loss"], label = "validation")

  # labeling the plot
  plt.gca().set(xlabel = "epoch", ylabel = "loss")
  plt.legend()
  plt.title("Plot of Training vs Validation Loss")

  return figure



def plot_confusion_matrix(cm, class_names):
  """
  Creates a visually appealing representation of the confusion matrix using matplotlib

  ---------------

  Parameters
  cm: array of shape n x n,  a confusion matrix of integer classes
  class_names: string names of the integer classes

  ---------------

  returns a matplotlib figure containing the plotted confusion matrix.
  """

  # creating a figure
  figure = plt.figure(figsize=(8, 8))

  # plotting the confusion matrix
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

  # including a title and color bar and adjusting ticks
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  # labeling the plot
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  return figure


def plot_classified_images(images, true_labels,labels_pred, label_encoder, input = "correct"):
  """
  Plots the images that were correctly or incorrectly classified by the CNN model

  ---------------

  Parameters
  images: testing images that the model has not seen
  true_labels: actual labels in the testing set
  labels_pred: predicted labels from the model
  label_encoder: dictionary that maps an integer to each tumor/non-tumor class
  input: string that determines if we want correctly classified or incorrectly classified images
  ---------------

  returns a matplotlib figure with 25 correctly classified MRI images
  """
  # creating a figure
  figure = plt.figure(figsize=(20,20))

  # checks if the user wants to plot correctly or incorrectly classified images and gets the respective list of labels
  if input == "correct":
    labels = np.arange(len(true_labels))[true_labels == labels_pred]
  elif input == "incorrect":
    labels = np.arange(len(true_labels))[true_labels != labels_pred]
  
  # shuffling the labels so that we get various tumor types plotted
  np.random.shuffle(labels)

  # for loop that will plot 25 images
  for i in range(25):

    # getting the index of the ith correctly or incorrectly classified image
    idx = labels[i]

    # creating subplots and making asethetic adjustments
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # plotting the images
    plt.imshow(images[idx].reshape(images.shape[1],images.shape[1]))

    # labeling the x axis with predicted and actual label
    plt.xlabel("Predicted: " + str(label_encoder.get(labels_pred[idx])) +  "\nActual: " + str(label_encoder.get(true_labels[idx])))

  return figure



