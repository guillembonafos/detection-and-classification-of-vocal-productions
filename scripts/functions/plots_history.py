import matplotlib.pyplot as plt


def plot_metrics_binary(history):
    """
    Plot evolution of the metrics of the binary classification problem across the
    epochs.

    Parameters
    ----------
    history : pd.DataFrame
        DataFrame resuming the evolution of the metrics during the training

    Returns
    -------
    Plot the metrics.
    """
    metrics = ['output_binaire_loss', 'output_binaire_auc', 'output_binaire_precision', 'output_binaire_recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history['epoch'], history[metric], label='Train')
        plt.plot(history['epoch'], history['val_' + metric], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'output_binaire_loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'output_binaire_auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
    plt.show()


def plot_metrics_multi(history):
    """
    Plot evolution of the metrics of the multi classification problem across the
    epochs.

    Parameters
    ----------
    history : pd.DataFrame
        DataFrame resuming the evolution of the metrics during the training.

    Returns
    -------
    Plot the metrics.
    """
    metrics = ['output_multi_loss', 'output_multi_categorical_accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 1, n + 1)
        plt.plot(history['epoch'], history[metric], label='Train')
        plt.plot(history['epoch'], history['val_' + metric], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'output_multi_loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])

        plt.legend()
    plt.show()
