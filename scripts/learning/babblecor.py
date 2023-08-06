import scripts.functions.preprocessing as pp
from scripts.functions.metadata import meta_baby
from scripts.functions.hypermodel import WinWavTransferLearning
import os
import tensorflow as tf
import kerastuner as kt
import pandas as pd
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#####################
# Load the metadata #
#####################

# path to the recordings
fd = "/data/b17026985/donnees/donnees_babblecor"
# Import of metadata given with babblecor
metadonnees = pd.read_csv(os.path.join(fd, "private_metadata.csv"))
# Rename the column to match with those of other functions
metadonnees.rename(columns={"Answer": "label", "clip_ID": "fd"}, inplace=True)
# Absolute path
metadonnees["fd"] = fd + "/clips/clips_corpus/" + metadonnees["fd"]

#################
# Preprocessing #
#################

# All the metadata
meta_train, meta_test, meta_val = meta_baby(fd, metadonnees, data_augmentation=True)

# Preprocessing
train, steps_per_epoch = pp.preparation_data_set(meta_train, resample=True, batch_size=128, transfer_learning=True)
val = pp.preparation_data_set(meta_val, resample=False, batch_size=128, transfer_learning=True)

input_shape = next(iter(train.unbatch()))[0].shape

train = train.shuffle(10000)

############
# Training #
############

# Training and optimisation
hypermodel = WinWavTransferLearning(input_shape=input_shape, n_labels=5)
hp = kt.HyperParameters()

tuner = kt.tuners.bayesian.BayesianOptimization(hypermodel=hypermodel,
                                                hyperparameters=hp,
                                                objective=kt.Objective("val_loss", direction="min"),
                                                max_trials=20,
                                                num_initial_points=5,
                                                tune_new_entries=True,
                                                project_name='babblecor')

# Instanciation of callbacks
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                             patience=20, verbose=1,
                                             restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(), 'babblecor/cp.hdf5'),
                                                monitor='val_loss', mode='min',
                                                save_best_only=True, verbose=1)

history = tf.keras.callbacks.CSVLogger(os.path.join(os.getcwd(), 'babblecor/train.csv'),
                                       separator=",", append=False)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                   patience=5, verbose=1)

callbacks = [earlystop, checkpoint, history, lr_schedule]

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train = train.with_options(options)
val = val.with_options(options)

print("Start of the learning")
start = datetime.now()
# steps_per_epoch necessary because infinite ds
tuner.search(train, epochs=1000, validation_data=val, callbacks=callbacks,
             steps_per_epoch=steps_per_epoch)
duree = datetime.now() - start

###########
# Results #
###########

tuner.results_summary()

best_hps = tuner.get_best_hyperparameters()[0]

print(f"""
Duration of the learning is {duree}. \n
Hperparameters research is over. \n
The optimal number of layers is {best_hps.get('n_layers')}. \n
The optimal learning rate is {best_hps.get('learning_rate')}. \n
The optimal drop-out rate is {best_hps.get('p_dropout')}.
""")
