import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn as sk
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def matrice_confusion(ds, model, meta, name_save=None, prob=False):
    """
    Make the prediction of the ds dataset by the model and compute the confusion
    matrices. Thanks to meta, we have access to the names of the labels to make
    the graphs. We can print the relative values instead of the absolute by specifying
    prob=True.

    Parameters
    ----------
    ds: tf.dataset
        Dataset with the recordings to predict, for which we want to construct
        the confusion matrices. The ds outputs by the function pp.preparation_data_set
        fits for.
    model: tf.model
        Model trained that will be used to do the prediction of the data in ds.
    meta: pd.DataFrame
        Metadata of the recordings in ds. Used to recover the label names for
        the graph of the confusion matrices.
    name_save: list
        List of two strings, giving the name of the file if we want to save the
        confusion matrices. They will be saved in ./images/name_save[0] and
        ./images/name_save[1]. If None, the plots are not saved.
    prob: bool
        Should we compute the relative frquency or give the absolute correct and
        incorrect classification in the graph. If True, compute the relative.

    Returns
    -------
    mc_binaire: plt.plot
        Confusion matrix of the binary classification problem, classification of
        noise againts vocalization.
    mc_multi_voc: plt.plot
        Confusion matrix of the multi classification problem, classification of
        the vocalization between the label of vocalization the species produces.
    """
    sns.set()

    # We start the timer
    start = datetime.now()
    # We do the prediction
    ypred = model.predict(ds)
    # Time to do the prediction
    duree_pred = datetime.now() - start
    print('Prediction time of the data by the model:', duree_pred)

    # List to keep the true labels
    y_binaire = []
    y_multi_complet = []
    # Iteration of the ds to take the labels (binary and multi)
    for audio, label, weight in ds:
        y_binaire.append(label['output_binaire'].numpy())
        y_multi_complet.append(label['output_multi'].numpy())
    # Flat the list (prediction is done per batch)
    y_binaire = [item for sublist in y_binaire for item in sublist]
    y_multi_complet = [item for sublist in y_multi_complet for item in sublist]
    # Code corresponding to the true label
    y_binaire = np.array(y_binaire)
    y_multi_complet = np.argmax(y_multi_complet, axis=1)

    # Coding of the labels
    le_binaire = sk.preprocessing.LabelEncoder()
    le_binaire.fit_transform(meta["label_binaire"])
    le_multi = sk.preprocessing.LabelEncoder()
    le_multi.fit_transform(meta["label"])

    # Results of the prediction
    ypred_binaire = ypred[0]
    ypred_multi = np.argmax(ypred[1], axis=1)
    n_classe = ypred[1].shape[1]

    # Make the correspondance to do not count the noise label in the multi classifcation
    # problem
    # The prediction of the recordings which are not voc
    ypred_multi_bruit = ypred_multi[y_binaire == 0]
    y_multi_complet_bruit = y_multi_complet[y_binaire == 0]
    # Prediction of recordings which are voc
    ypred_multi_vocs = ypred_multi[y_binaire == 1]
    y_multi_complet_vocs = y_multi_complet[y_binaire == 1]

    # Compute the confusino matrices
    # For the binary problem
    mc_binaire = tf.math.confusion_matrix(y_binaire, ypred_binaire > 0.5, num_classes=2, weights=None, dtype=tf.dtypes.int32, name=None)
    # For the multi problem
    mc_multi_voc = tf.math.confusion_matrix(y_multi_complet_vocs, ypred_multi_vocs, num_classes=n_classe, weights=None, dtype=tf.dtypes.int32, name=None)
    mc_multi_bruit = tf.math.confusion_matrix(y_multi_complet_bruit, ypred_multi_bruit, num_classes=n_classe, weights=None, dtype=tf.dtypes.int32, name=None)

    if prob is True:
        mc_binaire = np.round((mc_binaire.numpy().T / mc_binaire.numpy().sum(axis=1)).T * 100)
        mc_multi_voc = np.round((mc_multi_voc.numpy().T / mc_multi_voc.numpy().sum(axis=1)).T * 100)
        mc_multi_bruit = np.round((mc_multi_bruit.numpy().T / mc_multi_bruit.numpy().sum(axis=1)).T * 100)
    else:
        pass

    # Plot of confusion matrices
    # Matrix of the binary problem
    plt.figure()
    if prob is True:
        sns.heatmap(mc_binaire.astype(int), annot=True, fmt='d', xticklabels=le_binaire.classes_, yticklabels=le_binaire.classes_, cmap="Blues")
    else:
        sns.heatmap(mc_binaire, annot=True, fmt='d', xticklabels=le_binaire.classes_, yticklabels=le_binaire.classes_, cmap="Blues")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    if prob is True:
        plt.title('Predicted given the true')
    else:
        plt.title('Confusion matrix')
    if name_save is None:
        pass
    else:
        plt.savefig(os.path.join(os.getcwd(), "images", name_save[0]), bbox_inches="tight")
    plt.show()
    # Matrix of the multi problem
    plt.figure()
    if prob is True:
        sns.heatmap(mc_multi_voc.astype(int), annot=True, fmt='d', xticklabels=le_multi.classes_, yticklabels=le_multi.classes_, cmap="Blues")
    else:
        sns.heatmap(mc_multi_voc, annot=True, fmt='d', xticklabels=le_multi.classes_, yticklabels=le_multi.classes_, cmap="Blues")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    if prob is True:
        plt.title('Predicted given the true')
    else:
        plt.title('Confusion matrix')
    if name_save is None:
        pass
    else:
        plt.savefig(os.path.join(os.getcwd(), "images", name_save[1]), bbox_inches="tight")
    plt.show()

    return mc_binaire, mc_multi_voc
