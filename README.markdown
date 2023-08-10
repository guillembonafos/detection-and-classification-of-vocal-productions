---
jupyter:
  kernelspec:
    display_name: detection-and-classification-of-vocal-productions-main
    language: python
    name: detection-and-classification-of-vocal-productions-main
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.2
  nbformat: 4
  nbformat_minor: 4
---

``` python
#import scripts.functions.preprocessing as pp
#from scripts.functions.metadata import metadata_longform_papio, meta_papio
#from scripts.functions.hypermodel import WinWavTransferLearning
#from scripts.functions.segmentation import segmentation, df_pred, wav_creation
import os
import tensorflow as tf
import kerastuner as kt
import pandas as pd
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

``` python
%run ./scripts/functions/preprocessing.py
%run ./scripts/functions/metadata.py
%run ./scripts/functions/hypermodel.py
%run ./scripts/functions/segmentation.py
%run ./scripts/functions/data_augmentation.py
```

We present here an exemple to replicate the model used in the paper
<https://arxiv.org/abs/2302.07640>

The labeled data, as well as the longform audio recordings used as
exemple, and all the vocalizations extracted during the month, are
available at <https://doi.org/10.5281/zenodo.7963124>

The folder structure should be as follows: in the working directory,

-   a folder *data*. In this folder,
    -   a folder *vocalizations*, in which we have the labeled data,
    -   a folder *data_augmentation*, in which we save the modified
        vocalizations,
        -   a folder *test*, for the modified recordings of the test
            partition,
        -   a folder *train*, for the modified recordings of the train
            partition,
        -   a folder *val*, for the modified recordings of the
            validation partition.\
            For each one, there are three folders, one per condition of
            modification of the original recordings:
            -   a folder *noise*,
            -   a folder *tonality*,
            -   a folder *speed*,
    -   a folder *longform_recordings*, in which we have the long form
        audio recordings,
    -   a folder *segmentation_papio*, in which we will save the
        vocalizations detected.
-   a folder *scripts*. In this folder,
    -   a folder *functions*, in which we have all the functions used
        during the pipeline
    -   a folder *learning*, for the scripts used for the training of
        the models presentated in the paper,
    -   a folder *prediction*, for the scripts used for the prediction
        of all the long form audio recordings presented in the paper,
        that ouputs the data of the baboon and baby vocalizations.

The gitlab repository is already organized that way. The data available
on zenodo are organized that way. The libraries to install can be found
in the document requirements.txt

We present here an exemple to replicate the model using a subset of the
baboon recordings. We cannot provide all the month for legal reasons. We
put 2 hours accessible, as well as the labeled data set used in the
paper. We show how to train a model from the labeled dataset and how to
use it for the segmentation of these two hours.\
The total output of the segmentation of the month is available on
zenodo.

For the baby data, none of the long form audio recordings are available,
nor the output of the segmentation, for leagal reasons. The labeled
dataset is BabbleCor and can be found here <https://osf.io/rz4tx/>

# Learning

We start loading the metadata of the labeled data for the learning.

``` python
meta_train, meta_val, meta_test = meta_papio(os.getcwd(), data_augmentation=False, weighting_sampling=False)
```

``` python
wav_noise = [os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_01-05-2017_09h-10h_sans_vocs.wav"),
            os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_03-05-2017_11h-12h_sans_vocs.wav"),
            os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_05-05-2017_16h-17h_sans_vocs.wav"),
            os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_29-04-2017_18h-19h_sans_vocs.wav")]
```

Rubberband-cli need to be installed on the computer to perform data augmentation.

``` python
augmentation_data(meta_train, path_saved=os.path.join(os.getcwd(), "data/data_augmentation/train"),
                 wav_noise=wav_noise, n_noise=1,
                 n_tonality=5, low_tonality=-4, high_tonality=4,
                 n_speed=5, low_speed=-0.3, high_speed=0.3)
augmentation_data(meta_val, path_saved=os.path.join(os.getcwd(), "data/data_augmentation/val"),
                 wav_noise=wav_noise, n_noise=1,
                 n_tonality=5, low_tonality=-4, high_tonality=4,
                 n_speed=5, low_speed=-0.3, high_speed=0.3)
```

Then, we prepare the data. This is done through the creation of a
dataset, using the metadata of the labeled data. The recordings are
loaded and resampled at 16 KhZ, in mono. We create independant frames
through an 80% overlapping 1-second window. We use a resampling strategy
to have an uniform distribution among classes during the learning.
Because we use transfer-learning from YamNet, the frames are mapped to a
log-mel spectrogram. Data augmentation is done before, not during the
learning, because we do not expect to have so much labeled data. Thus,
we can gain time during the learning without being too expensive in term
of memory.

``` python
train, steps_per_epoch = preparation_data_set(meta_train, resample=True, batch_size=32, transfer_learning=True)
val = preparation_data_set(meta_val, resample=False, batch_size=32, transfer_learning=True)

input_shape = next(iter(train.unbatch()))[0].shape

train = train.shuffle(1000)
```

We instantiate the hypermodel and the hyperparameters, as well as the
callbacks. The values set here are for the exemple and can be changed
and increase for a \"true\" learning.

``` python
hypermodel = WinWavTransferLearning(input_shape=input_shape, n_labels=6)
hp = kt.HyperParameters()

tuner = kt.tuners.bayesian.BayesianOptimization(hypermodel=hypermodel,
                                               hyperparameters=hp,
                                               objective=kt.Objective("val_loss", direction="min"),
                                                # increase to have more searching iterations. 
                                                # Set to 2 here for the exemple
                                                max_trials=2,
                                                num_initial_points=1,
                                                tune_new_entries=True,
                                                project_name="exemple")

earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, 
                                             patience=5, verbose=1, 
                                             restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(), "exemple/cp.hdf5"),
                                               monitor="val_loss", mode="min", 
                                                save_best_only=True, verbose=1)

history = tf.keras.callbacks.CSVLogger(os.path.join(os.getcwd(), "exemple/train.csv"),
                                      separator=",", append=False)


lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                                  patience=2, verbose=1)

callbacks = [earlystop, checkpoint, history, lr_schedule]
```

We can start the learning.

``` python
# Disable AutoShard
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train = train.with_options(options)
val = val.with_options(options)

print("Start of the learning")
start = datetime.now()
tuner.search(train, epochs=20, validation_data=val, callbacks=callbacks,
            steps_per_epoch=steps_per_epoch)
delta_training = datetime.now() - start
```


# Prediction

The model has been learned on the labeled data and now can be used to
detect vocalizations in long form audio recordings. We start loading the
metadata of the longform audio recordings as well as their length and we
prepare the data to be processed by the model.

``` python
meta_longform = metadata_longform_papio(os.path.join(os.getcwd(), "data/longform_recordings"), 
                                       length=True)

ds = preparation_longform_papio(meta_longform, batch_size=32, transfer_learning=True)
```

We take the best model of the optimization process and we use it to find
the segments of vocalizations in the recordings.

``` python
model = tuner.get_best_models()[0]
model.summary()
```

``` python
start = datetime.now()
y = model.predict(ds)
delta_pred = datetime.now() - start
print("Duration prediction:", delta_pred)
```


# Segmentation

Once we learned the model and used it to find the segments of
vocalizations in the longform audio recordings, we extract the
information. First, we create txt files, one per recordings, in which we
have the number of vocalizations found with the time in the recordings.

``` python
segmentation(meta_longform, y, baby=False)
```

Then, we take the information that we have in the txt files to create
the wav files, one per vocalization.

``` python
wav_creation(fd=os.path.join(os.getcwd(), "data/longform_recordings"), baby=False)
```

We create a dataframe in which we have more information for each
vocalization the model detected (the day, the hour, the duration, the
probability of each label).

``` python
df = df_pred(meta_longform, y)
```
