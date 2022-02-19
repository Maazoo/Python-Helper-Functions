import matplotlib.pyplot as plt
import zipfile
import os
import tensorflow as tf
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def plot_history(history):

    """
    Function to plot the Training and Validation loss and accuracy using matplotlib.

    Args: 
        history: tensorflow model history object.
    """

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(1,len(history.history["loss"])+1)

    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("Accuracy")


def unzip_data(filename):
    """
    Unzips filename into current working directory

    Args:
        filname: a filepath to a target zip folder to be unzipped


    """
    zip_ref = zipfile.ZipFile(filename,"r")
    zip_ref.extractall()
    zip_ref.close()

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.
  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"
  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%H:%M:%S-%d.%m.%Y")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


def compute_metrics(y_true,y_pred):
  """
  Computes precision, recall, f1 score and accuracy.
  Args: y_true: ground truth labels
        y_pred: predicted labels  
  Returns: results: a dictionary of precision, recall, f1 score and accuracy"""
  accuracy = accuracy_score(y_true,y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(y_true,y_pred,average="weighted")
  results = {"accuracy":accuracy,
             "precision":precision,
             "recall":recall,
             "f1":f1}
  return results
