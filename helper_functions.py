import matplotlib.pyplot as plt
import zipfile

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