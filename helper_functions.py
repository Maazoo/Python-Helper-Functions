import matplotlib.pyplot as plt

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

