import matplotlib.pyplot as plt


def plot_loss(history):

    plt.title("Training Loss")
    plt.plot(history.history['loss'])
    plt.show()

    plt.title("Validation Loss")
    plt.plot(history.history['val_loss'])
    plt.show()


def plot_accuracy(history):

    plt.figure(figsize=(5,5))

    plt.plot(
        history.history['accuracy']
    )

    plt.plot(
        history.history['val_accuracy']
    )

    plt.title("Accuracy Curve")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(
        ['train', 'validation']
    )

    plt.show()
