import scripts.functions.preprocessing as pp
from scripts.functions.metadata import metadata_longform_papio, meta_papio
from scripts.functions.hypermodel import WinWavTransferLearning
from scripts.functions.segmentation import segmentation, df_pred, wav_creation
import os
import tensorflow as tf
import kerastuner as kt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn as sk
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Metadata
meta_longform = metadata_longform_papio()
# If we need the length of the files. Much longer!
meta_longform = metadata_longform_papio(length=True)
# On enregistre pour ne pas avoir à le relancer une autre fois
meta_longform.to_csv(os.path.join(os.getcwd(), 'data/meta_continu_papio.csv'))
# Chargement direct
meta_longform = pd.read_csv(os.path.join(os.getcwd(), 'data/meta_continu_papio.csv'))

############################
# Prediction of recordings #
############################

ds = pp.preparation_longform_papio(meta_longform, batch_size=32, transfer_learning=True)
input_shape = next(iter(ds.unbatch()))[0].shape

# Take-back the model
hm = WinWavTransferLearning(input_shape=input_shape, n_labels=6)
hp = kt.HyperParameters()
tuner = kt.tuners.bayesian.BayesianOptimization(hypermodel=hm,
                                                hyperparameters=hp,
                                                objective=kt.Objective("val_loss", direction="min"),
                                                max_trials=1,
                                                num_initial_points=1,
                                                tune_new_entries=True,
                                                overwrite=False,
                                                project_name='papio')

modele = tuner.get_best_models()[0]
modele.summary()

# Prediction
start = datetime.now()
y = modele.predict(ds)
delta_pred = datetime.now() - start
print(delta_pred, "seconds")

# Save the results
with open("data/pred_papio.pickle", "wb") as fp:
    pickle.dump(y, fp)

# Load if necessary
with open("data/pred_papio.pickle", 'rb') as handle:
    y = pickle.load(handle)

################
# Segmentation #
################

# For each recording, we write the detected vocalization in a txt file
segmentation(meta_longform, y, baby=False)

# From the txt files created in the directory, we create the wav files of the
# vocalizations
wav_creation(fd="/media/guillhem/BackUParnaud/LOG_DATA", baby=False)

###########################################
# Description of the volizations detected #
###########################################

# Summary of the prediction in a df
df = df_pred(meta_longform, y)

# save info
df.to_csv(os.path.join(os.getcwd(), 'data/df_composition_papio.csv'))
df = pd.read_csv(os.path.join(os.getcwd(), 'data/df_composition_papio.csv'))

# Total duration in seconds
tps_sec = meta_longform["len"].sum()/16000
# Minutes
tps_min = tps_sec/60
# Hours
tps_h = tps_min/60
print("Recordings to process last", tps_sec, "seconds, i.e.,", tps_min, "minutes or", tps_h, "hours")

# Code of the classes
meta_train, meta_val, meta_test = meta_papio(os.getcwd(), data_augmentation=False, weighting_sampling=False)
le = sk.preprocessing.LabelEncoder()
le.fit(meta_train["label"])
le.classes_
df["classe"] = le.inverse_transform(df["type_voc"].astype(int))

#####################################
# Mean duration of the vocalization #
#####################################

# Describe
df["duree_voc"].describe()
# Global mean duration
df["duree_voc"].mean()
# Per class duration
df.groupby("classe")["duree_voc"].mean()
# Effective per class
df.groupby("classe")["type_voc"].count()

# Total duration of vocalization
df["duree_voc"].sum()/60/60

